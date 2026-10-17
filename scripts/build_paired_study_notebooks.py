"""Generate the three study notebooks with a fixed, committed runner revision."""
import argparse
import json
from pathlib import Path
import textwrap


def md(text):
    return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip().splitlines(True)}


def code(text):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':(textwrap.dedent(text).strip()+'\n').splitlines(True)}


def bootstrap(revision):
    return code('''
        from pathlib import Path
        import sys, os, json, subprocess, importlib.util
        IN_COLAB = "google.colab" in sys.modules
        CODE_VERSION = "REVISION"
        if any(name == "uncle" or name.startswith("uncle.") for name in sys.modules):
            raise RuntimeError("Restart the runtime before loading this fixed code revision.")
        if IN_COLAB:
            REPO = Path("/content/cark-paired-study")
            if not REPO.exists():
                subprocess.run(["git", "clone", "https://github.com/sumitasthana/PATH.git", str(REPO)], check=True)
            if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
                raise RuntimeError("Checkout has edits. Use a fresh runtime.")
            subprocess.run(["git", "-C", str(REPO), "fetch", "origin"], check=True)
            subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", CODE_VERSION], check=True)
        else:
            REPO = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "uncle/task_partition.json").exists()), None)
            if REPO is None:
                raise FileNotFoundError("Open this notebook from a CARK checkout.")
            actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
            implementation = ["uncle", "pyproject.toml", "requirements.txt"]
            dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--", *implementation], text=True).strip()
            # A later documentation-only commit may contain the same fixed implementation.
            differences = subprocess.check_output(["git", "-C", str(REPO), "diff", CODE_VERSION, actual, "--", *implementation], text=True).strip()
            if dirty or differences:
                raise RuntimeError("Use a clean checkout of CODE_VERSION, or open in Colab. Local training edits are preserved.")
        for package in ("boto3", "tqdm", "matplotlib"):
            if importlib.util.find_spec(package) is None:
                if not IN_COLAB:
                    raise ImportError("Install the storage and extras dependencies before running this notebook.")
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], check=True)
        sys.path.insert(0, str(REPO))
        from uncle.storage import R2Store
        from uncle.paired_study import StudyStore, make_plan, jobs, inspect_queue, run_queue, review
        import torch
        print("Fixed implementation revision:", CODE_VERSION)
        print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    '''.replace('REVISION', revision))


def connection():
    return code('''
        def secret(name):
            value = os.environ.get(name)
            if not value and IN_COLAB:
                from google.colab import userdata
                value = userdata.get(name)
            if not value:
                raise RuntimeError("Missing secret: " + name)
            return value
        store = R2Store.connect(secret("R2_ENDPOINT"), BUCKET,
            secret("R2_ACCESS_KEY_ID"), secret("R2_SECRET_ACCESS_KEY"))
        store.progress = True
        CACHE = REPO / "outputs/paired_study" / STUDY_ID
        CACHE.mkdir(parents=True, exist_ok=True)
        print("Connected to R2 bucket:", BUCKET)
    ''')


def load_plan():
    return code('''
        PREFIX = "uncle/paired_generalization/" + STUDY_ID
        path = store.download(PREFIX + "/study.json", CACHE / "cache")
        plan = json.loads(path.read_text())
        if plan["code_version"] != CODE_VERSION:
            raise ValueError("This study requires a different fixed code revision. Use its recorded notebook revision.")
        from uncle.learning_diagnostic import file_hash
        if file_hash(REPO / "uncle/task_partition.json") != plan["partition_sha256"]:
            raise ValueError("Class partition differs from the saved study.")
        storage = StudyStore(store, plan, CACHE, cleanup_completed=CLEAN_COMPLETED_RESUME_SLOTS)
        rows = inspect_queue(plan, storage, progress=True)
        if globals().get("SELECTED_JOB_IDS"):
            rows = [row for row in rows if row["id"] in SELECTED_JOB_IDS]
            print("Showing selected jobs only; the full saved study remains unchanged.")
        from collections import Counter
        print("Queue:", dict(Counter(row["status"] for row in rows)))
        for row in rows:
            print(row["status"].ljust(10), row["id"], row["active"] or "")
    ''')


def resilient_connection():
    cell = connection()
    extra = code('''
        # Transport-only fix: preserve the saved training revision and resume contract.
        from boto3.s3.transfer import TransferConfig
        from boto3.exceptions import S3UploadFailedError
        from botocore.exceptions import ClientError
        import time
        transfer_config = TransferConfig(multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024, max_concurrency=1,
            use_threads=False, preferred_transfer_client="classic")
        original_upload_file = store.client.upload_file
        def upload_with_multipart_retry(*args, **kwargs):
            kwargs["Config"] = transfer_config
            for attempt in range(1, 4):
                try:
                    return original_upload_file(*args, **kwargs)
                except (S3UploadFailedError, ClientError) as error:
                    error_code = getattr(error, "response", {}).get("Error", {}).get("Code")
                    invalid_part = error_code == "InvalidPart" or (
                        isinstance(error, S3UploadFailedError) and "(InvalidPart)" in str(error))
                    if not invalid_part or attempt == 3:
                        raise
                    print("R2 InvalidPart: restarting the full upload, attempt", attempt + 1,
                        "of 3. Saved progress has not advanced.", flush=True)
                    callback_owner = getattr(kwargs.get("Callback"), "__self__", None)
                    if callback_owner is not None and hasattr(callback_owner, "reset"):
                        callback_owner.reset()
                    time.sleep(2 * attempt)
        store.client.upload_file = upload_with_multipart_retry
        print("R2 uploads: sequential 64 MiB parts; at most three InvalidPart attempts.")
    ''')
    cell['source'] += extra['source']
    return cell


def guarded_setup(cell):
    source = ''.join(cell['source'])
    wrapped = 'try:\n' + textwrap.indent(source, '    ')
    wrapped += '''except BaseException:
    import traceback
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    if "dashboard" in globals():
        dashboard.finish("Setup failed; see traceback")
        dashboard.restore()
    if RELEASE_GPU_WHEN_DONE and IN_COLAB:
        from google.colab import runtime
        print("Setup failed before training. Releasing the GPU runtime.")
        runtime.unassign()
    raise
'''
    return code(wrapped)


def selected_queue():
    return code('''
        # Select execution only. The immutable plan and job contracts stay unchanged.
        import types
        from uncle.paired_study import inspect_queue as full_inspect_queue
        def selected_inspect_queue(saved_plan, saved_storage, *, progress=True):
            all_jobs = jobs(saved_plan)
            known = {job["id"]: job for job in all_jobs}
            if not SELECTED_JOB_IDS or len(set(SELECTED_JOB_IDS)) != len(SELECTED_JOB_IDS):
                raise ValueError("Select a nonempty list of distinct job IDs.")
            if set(SELECTED_JOB_IDS) - known.keys():
                raise ValueError("Selected job is outside the saved study.")
            selected = set(SELECTED_JOB_IDS)
            if any(known[key]["source"] and known[key]["source"] not in selected
                   for key in selected):
                raise ValueError("Include the source job for each selected continuation.")
            rows = full_inspect_queue(saved_plan, saved_storage, progress=progress)
            by_id = {row["id"]: row for row in rows}
            return [by_id[key] for key in SELECTED_JOB_IDS]
        # Give only the session scheduler a filtered queue, without changing module globals.
        scheduler_globals = dict(run_queue.__globals__, inspect_queue=selected_inspect_queue)
        if "dashboard" in globals():
            dashboard.configure(plan, storage, jobs(plan))
            original_run_job = run_queue.__globals__["run_job"]
            original_request = original_run_job.__globals__["run_request"]
            def display_request(trainer, action, task, *args, **kwargs):
                dashboard.request(action, task)
                return original_request(trainer, action, task, *args, **kwargs)
            job_globals = dict(original_run_job.__globals__, run_request=display_request)
            display_job = types.FunctionType(original_run_job.__code__, job_globals,
                "display_job", original_run_job.__defaults__, original_run_job.__closure__)
            display_job.__kwdefaults__ = dict(original_run_job.__kwdefaults__ or {})
            def run_display_job(saved_plan, job, *args, **kwargs):
                dashboard.start_job(job)
                kwargs["on_boundary"] = dashboard.saved
                return display_job(saved_plan, job, *args, **kwargs)
            scheduler_globals["run_job"] = run_display_job
        run_selected_queue = types.FunctionType(run_queue.__code__, scheduler_globals,
            "run_selected_queue", run_queue.__defaults__, run_queue.__closure__)
        run_selected_queue.__kwdefaults__ = dict(run_queue.__kwdefaults__ or {})
        selected_rows = selected_inspect_queue(plan, storage, progress=True)
        print("Selected deadline queue:")
        for row in selected_rows:
            print(row["status"].ljust(10), row["id"], row["active"] or "")
        print("All other study jobs remain saved but will not run in this session.")
    ''')


def compact_progress():
    return code((Path(__file__).with_name('paired_progress.py')).read_text(encoding='utf-8'))


def prepare(revision):
    return [
        md('''
        # 18. Prepare the paired generalization study

        Use a CPU runtime. This notebook fixes the experiment matrix, inspects saved progress,
        and publishes the study configuration to R2. It trains no models and uploads no images.

        The default study uses three seeds, three earlier learning orders, three forgotten tasks,
        and incoming task 15. Each target occurs at positions 1, 4, and 8 across the orders.
        Nine source models and nine shared learning-only controls support 27 paired comparisons.
        There are 45 executable jobs: nine sources and 36 branches.

        These are diagnostic settings from experiment 06. They are not an exact paper reproduction
        and do not include a recovery probe. Notebooks 16 and 17 remain separate.
        Source histories are trained fresh with one fixed configuration, including beta 0.1
        for all eight lessons. They do not recreate the earlier mixed-setting checkpoint chain.
        '''),
        md('## 1. Load the fixed code'), bootstrap(revision),
        md('## 2. Choose the study'), code('''
        STUDY_ID = "paired_generalization_v1"
        BUCKET = "uncle-experiments"
        MODE = "CHECK"  # CHECK reads only; PREPARE publishes the configuration.
        if MODE not in ("CHECK", "PREPARE"):
            raise ValueError("Choose CHECK or PREPARE.")
        if torch.cuda.is_available():
            raise RuntimeError("Switch to a CPU runtime for preparation.")
        plan = make_plan(CODE_VERSION, REPO / "uncle/task_partition.json", STUDY_ID)
        # Edit the matrix before PREPARE, using a new study ID for changed settings.
        # For a smaller first study, keep order_01, seed 0, and forget task 3.
        # Use STUDY_ID = "paired_generalization_pilot_v1" for that smaller matrix.
        # plan["orders"] = {"order_01": plan["orders"]["order_01"]}
        # plan["seeds"] = [0]
        # plan["forget_tasks"] = ["3"]
        # plan["keep_post_unlearning_checkpoint"] = True  # retain these for selected follow-up work
        # plan["keep_branch_checkpoints"] = True  # retain all branch finals for later model inspection
        from uncle.paired_study import validate_plan
        validate_plan(plan)
        print(json.dumps(plan, indent=2))
        '''),
        md('## 3. Inspect compute and storage requirements'), code('''
        queue = jobs(plan)
        from collections import Counter
        counts = Counter(job["kind"] for job in queue)
        paired_count = counts["unlearned"]
        learning_lessons = sum(action == "learn" for job in queue for action, task in job["operations"])
        approx_model_bytes = 538_000_000  # approximate observed model size, not a storage limit
        retained_models = counts["source"]
        if plan["keep_branch_checkpoints"]:
            retained_models += counts["control"] + counts["unlearned"]
        if plan["keep_post_unlearning_checkpoint"]:
            retained_models += paired_count
        print("Work units:", dict(counts))
        print("Paired comparisons:", paired_count)
        print("Learning lessons:", learning_lessons, "Unlearning requests:", paired_count)
        print("Approximate final model storage:", round(retained_models * approx_model_bytes / 1e9, 2), "GB")
        print("Two active resume slots add roughly 4.3 GB for a large job; actual size is measured when saved.")
        print("Runtime is unknown until the first GPU sessions measure training, evaluation, and transfers.")
        print("Resume saves include Adam and the original protection snapshot, so they exceed final model size.")
        print("Sources remain available. Branch models are deleted after verified completion; reports and hashes remain.")
        print("Temporary slots are removed only after the final model and report are verified.")
        print("No existing experiment folders are deleted. No dataset images are uploaded.")
        '''),
        md('''
        ## 4. Connect to R2

        Supply `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY` through Colab Secrets
        or environment variables. Their values are never printed or saved in the notebook.
        Use only one active writer per study.
        '''), connection(),
        md('## 5. Check or publish the configuration'), code('''
        storage = StudyStore(store, plan, CACHE)
        existing = storage.read(storage.prefix + "/study.json")
        if existing is not None and existing != plan:
            raise ValueError("This study ID already has different settings. Restore them or choose a new ID.")
        if MODE == "PREPARE":
            storage.prepare()
            print("Study configuration uploaded and verified. No training started.")
        elif existing is None:
            print("New study. Inspect this matrix, then use MODE = PREPARE to save it.")
        else:
            print("Saved study matches this configuration.")
        rows = inspect_queue(plan, storage, progress=True)
        print("Queue:", dict(Counter(row["status"] for row in rows)))
        for row in rows:
            print(row["status"].ljust(10), row["id"], row["active"] or "")
        print("Next: open notebook 19 with the same STUDY_ID in a GPU runtime.")
        '''),
    ]


def gpu(revision):
    return [
        md('''
        # 19. Run or resume the paired generalization study

        The existing paired_generalization_v2 study is already prepared in R2.
        Change only SEED in section 2 and run all cells on A100. Training is enabled by default.
        Reopening with the same seed resumes verified unfinished work and skips completed jobs.
        A new seed trains its own source history. Do not rerun PREPARE for this existing study.

        Learning saves after each epoch. Unlearning saves at the configured step interval.
        A runtime loss can discard work since the last verified R2 boundary. An interrupted
        learning epoch restarts from its preceding boundary with the saved Adam, protection
        snapshot, random state, and data-order generator. It does not start with a fresh optimizer.

        The time budget is a soft limit: a boundary and its uploads may finish after the limit.
        Use a generous reserve. A GPU/software mismatch stops resume rather than silently changing
        the experiment. Resume support does not establish repeatability across different GPUs.
        '''),
        md('## 1. Load the fixed code'), bootstrap(revision),
        md('## 2. Set the session budget'), code('''
        STUDY_ID = "paired_generalization_v2"
        BUCKET = "uncle-experiments"
        SEED = 0  # Change only this value: 0, 1, or 2. Then run all cells.
        if type(SEED) is not int or SEED not in (0, 1, 2):
            raise ValueError("Choose SEED = 0, 1, or 2 from the saved study.")
        RUN_TRAINING = True
        MAX_JOBS = 4
        SELECTED_JOB_IDS = [
            f"sources/order_01/seed_{SEED}",
            f"controls/order_01/seed_{SEED}/learn_15",
            f"branches/order_01/seed_{SEED}/forget_3/learn_15",
            f"branches/order_01/seed_{SEED}/forget_14/learn_15",
        ]
        MAX_SESSION_MINUTES = 120
        SAVE_RESERVE_MINUTES = 15
        CLEAN_COMPLETED_RESUME_SLOTS = True
        RELEASE_GPU_WHEN_DONE = True
        DATA_ROOT = Path("/content/data/tiny-imagenet-200") if IN_COLAB else REPO / "data/tiny-imagenet-200"
        if not RUN_TRAINING:
            raise RuntimeError("Training is off. Set RUN_TRAINING = True when ready to run this session.")
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime for this notebook.")
        # These flags are recorded and must match on every resumed session.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("GPU:", torch.cuda.get_device_name(0))
        print("Session limit:", MAX_SESSION_MINUTES, "minutes; reserve:", SAVE_RESERVE_MINUTES)
        '''),
        md('## 3. Connect and recover the queue'), compact_progress(),
        guarded_setup(resilient_connection()), guarded_setup(load_plan()),
        guarded_setup(selected_queue()),
        md('''
        ## 4. Run the next work units

        Change only SEED in section 2, then run all cells on A100. This queue uses
        order 01 and the selected seed: source, L15-only control, U3 then L15,
        followed by U14 then L15. Completed jobs are reused. Other orders and forget
        targets are deferred. Keep the same study ID and saved plan.
        MAX_JOBS = 4 allows all four jobs if the session budget permits; an unfinished
        job stops the session and resumes next time with the same seed. A new seed
        needs its own eight-lesson source. One session may not finish it.

        Tiny ImageNet is downloaded to local runtime storage only when needed, with a download
        progress bar. It is reused across the session. Evaluation, learning, unlearning, R2 transfers,
        and the study queue also have progress bars.

        Every progress pointer is published after its checkpoint and report are verified.
        Source models remain available for every branch. By default, branch final models are
        deleted after the completed report and model verification receipt are durably saved.
        Immediate post-unlearning models are not retained by default. All reports, settings,
        hashes, and figures remain. Unfinished jobs retain resumable model and optimizer state.
        Cleanup touches only this study's completed-job model files and temporary slots.
        It never deletes historical experiment files.
        '''),
        code('''
        from uncle.streams import build_tasks
        dashboard.attach_here()
        task_cache = {}
        def tasks_factory(job):
            dashboard.start_job(job)
            wanted = tuple(sorted(set(plan["orders"][job["order"]]) |
                ({job["new_task"]} if job["source"] else set())))
            if wanted not in task_cache:
                # Dataset preparation occurs inside the timed session, not before its budget.
                task_cache[wanted] = build_tasks(root=DATA_ROOT,
                    partition_path=REPO / "uncle/task_partition.json", partition_seed=42,
                    include=wanted, download=True)
            return task_cache[wanted]

        session = None
        try:
            session = run_selected_queue(plan, storage, tasks_factory, max_jobs=MAX_JOBS,
                max_minutes=MAX_SESSION_MINUTES, reserve_minutes=SAVE_RESERVE_MINUTES,
                device="cuda", progress=True)
            print(json.dumps(session, indent=2))
            print("Session progress saved. Reopen this notebook with the same STUDY_ID to continue.")
        except BaseException as error:
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            print("Session interrupted:", type(error).__name__)
            print("Verified earlier R2 boundaries remain available. Do not change the study settings to resume.")
            raise
        finally:
            dashboard.finish("Session saved; run all again to continue" if session is not None
                else "Interrupted; resume from the latest verified R2 save")
            dashboard.restore()
            task_cache.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if RELEASE_GPU_WHEN_DONE and IN_COLAB:
                from google.colab import runtime
                print("Releasing the Colab runtime.")
                runtime.unassign()
        '''),
        md('''
        ## 5. Review on CPU

        Open notebook 20 to inspect completed pairs, interruption records, averages, and figures.
        It downloads reports rather than models. Do not occupy a GPU for analysis.
        '''),
    ]


def reporting(revision):
    return [
        md('''
        # 20. Review and average the paired study

        Use a CPU runtime. This notebook downloads verified reports, checks pairing, and shows
        every valid seed result alongside setting means and standard deviations. Invalid pairs
        remain visible and are excluded from averages. Pending and resumable work is listed.

        The overall average gives equal weight to settings with all planned seeds complete.
        A partial summary is explicitly labeled. Shared controls and retained tasks are correlated,
        so task rows and epoch measurements are not independent repetitions. Three seeds give
        limited precision. No confidence interval or information-erasure claim is produced.
        '''),
        md('## 1. Load the fixed code'), bootstrap(revision),
        md('## 2. Select the study'), code('''
        STUDY_ID = "paired_generalization_v2"
        BUCKET = "uncle-experiments"
        CLEAN_COMPLETED_RESUME_SLOTS = False  # Review is read-only until the final export cell.
        SAVE_SUMMARY_TO_R2 = True
        if torch.cuda.is_available():
            raise RuntimeError("Switch to a CPU runtime for analysis.")
        '''),
        md('## 3. Connect and inspect progress'), connection(), load_plan(),
        md('## 4. Read reports and compute paired effects'), code('''
        result = review(plan, storage, progress=True)
        print("Summary:", result["status"], "Valid pairs:", result["valid_pairs"], "/", result["expected_pairs"])
        print("Complete settings:", result["complete_settings"])
        print("Equal-setting means:", json.dumps(result["equal_setting_means"], indent=2))
        print("Negative new-task differences:", result["new_task_negative_pairs"])
        print("Within the fixed forgetting tolerance:", result["forgetting_within_tolerance_pairs"])
        print("Relapse above that tolerance:", result["relapse_pairs"])
        for row in result["invalid_pairs"]:
            print("INVALID PAIR:", row["branch_id"], row["pair_checks"])
        for row in result["unfinished_jobs"]:
            print("UNFINISHED:", row["status"], row["job"])
        for record in result["session_records"]:
            if record["status"] == "interrupted":
                print("INTERRUPTED SESSION:", record["session_id"], record.get("current_job"), record.get("error_type"))
        '''),
        md('## 5. Show seed results and individual retained-task changes'), code('''
        for setting in result["settings"]:
            print("\\n", setting["order"], "forget", setting["forget_task"], "then learn", setting["new_task"])
            print("Seeds:", setting["seeds"], "Complete setting:", setting["complete_setting"])
            for metric in ("new_task_difference", "retained_mean_difference", "largest_paired_retained_drop"):
                measurement = setting[metric]
                print(metric, "mean:", round(measurement["mean"], 3),
                    "SD:", measurement["sd"], "values:", measurement["values"])
        for row in result["comparisons"]:
            print(row["order"], "seed", row["seed"], "forget", row["forget_task"],
                "new-task difference:", row["new_task_difference"],
                "retained differences:", row["retained_differences"])
        print(result["limitation"])
        '''),
        md('## 6. Plot the paired effects'), code('''
        import matplotlib.pyplot as plt
        from datetime import datetime, timezone
        EXPORT = CACHE / "summaries" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        EXPORT.mkdir(parents=True, exist_ok=True)
        if result["settings"]:
            labels = [s["order"] + "/U" + s["forget_task"] + "/L" + s["new_task"] for s in result["settings"]]
            fig, axes = plt.subplots(2, 1, figsize=(max(9, len(labels)*0.9), 8), sharex=True)
            for axis, metric, title in zip(axes,
                ("new_task_difference", "retained_mean_difference"),
                ("New-task accuracy: B minus A", "Mean retained accuracy: B minus A")):
                for index, setting in enumerate(result["settings"]):
                    measurement = setting[metric]
                    values = measurement["values"]
                    offsets = [0.08*(i-(len(values)-1)/2) for i in range(len(values))]
                    axis.scatter([index+o for o in offsets], values, color="#245d99", label="Seed result" if index==0 else None)
                    axis.errorbar(index, measurement["mean"], yerr=measurement["sd"] or 0,
                        fmt="s", color="#b85b22", capsize=4, label="Mean and SD" if index==0 else None)
                axis.axhline(0, color="black", linewidth=0.8)
                axis.set_ylabel("Percentage points")
                axis.set_title(title)
                axis.legend()
            axes[-1].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            fig.suptitle("Paired study: " + result["status"] + " (SD describes seed variation)")
            fig.tight_layout()
            fig.savefig(EXPORT / "paired_effects.png", dpi=180)
            fig.savefig(EXPORT / "paired_effects.svg")
            plt.show()
        else:
            print("No valid completed pairs to plot yet.")
        '''),
        md('## 7. Save review artifacts'), code('''
        import csv
        from uncle.learning_diagnostic import _write_json
        _write_json(EXPORT / "summary.json", result)
        _write_json(EXPORT / "study.json", plan)
        fields = ["status", "order", "seed", "forget_task", "new_task", "control_id", "branch_id",
            "new_task_a", "new_task_b", "new_task_difference", "retained_mean_difference",
            "starting_target_accuracy", "after_unlearning_accuracy", "final_target_accuracy",
            "target_start_above_chance", "forgetting_within_tolerance", "relapse_above_tolerance",
            "largest_paired_retained_drop", "largest_temporary_retained_drop"]
        with (EXPORT / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["comparisons"])
        with (EXPORT / "retained_tasks.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["status", "order", "seed", "forget_task", "new_task", "retained_task", "difference_pp"])
            writer.writeheader()
            for row in result["comparisons"]:
                for task, difference in row["retained_differences"].items():
                    writer.writerow({**{key:row[key] for key in ("status","order","seed","forget_task","new_task")},
                        "retained_task":task, "difference_pp":difference})
        if SAVE_SUMMARY_TO_R2:
            from tqdm.auto import tqdm
            export_prefix = PREFIX + "/summaries/" + EXPORT.name
            for path in tqdm(sorted(EXPORT.iterdir()), desc="Save review artifacts", unit="file"):
                store.upload(path, export_prefix + "/" + path.name)
            print("Review artifacts uploaded and verified.")
            print("R2 review artifacts: s3://" + BUCKET + "/" + export_prefix + "/")
        else:
            print("R2 summary saving is disabled. These exports exist only in this runtime.")
        print("Local review artifacts:", EXPORT)
        print("No models trained, modified, or deleted by this review.")
        '''),
    ]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--code-version', required=True)
    args=parser.parse_args()
    if len(args.code_version)!=40 or any(c not in '0123456789abcdef' for c in args.code_version):
        raise ValueError('Use the full committed implementation SHA.')
    directory=Path(__file__).resolve().parents[1]/'notebooks'
    for name,cells in [
        ('18_paired_study_prepare.ipynb',prepare(args.code_version)),
        ('19_paired_study_run.ipynb',gpu(args.code_version)),
        ('20_paired_study_review.ipynb',reporting(args.code_version))]:
        notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},'colab':{'name':name,'provenance':[]}},'nbformat':4,'nbformat_minor':5}
        for index,cell in enumerate(cells):
            cell['id']=f'paired-{index:02d}'
        (directory/name).write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
        print(name)


if __name__=='__main__':
    main()
