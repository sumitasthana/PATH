import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.storage import R2Store, sha256


class Missing(Exception):
    response = {'Error': {'Code': '404'}}


class Files:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.downloads = 0
        self.corrupt_upload = False

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise Missing()
        data, metadata = self.objects[Key]
        return {'ContentLength': len(data), 'Metadata': metadata}

    def upload_file(self, path, bucket, key, ExtraArgs):
        data = Path(path).read_bytes()
        if self.corrupt_upload:
            data = b'X' * len(data)
        self.objects[key] = data, ExtraArgs['Metadata']
        self.uploads.append(key)

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key][0])}

    def download_file(self, bucket, key, path):
        self.downloads += 1
        Path(path).write_bytes(self.objects[key][0])

    def get_paginator(self, name):
        class Pages:
            def paginate(inner, Bucket, Prefix):
                for key in sorted(self.objects):
                    if key.startswith(Prefix):
                        yield {'Contents': [{'Key': key}]}
        return Pages()


class StorageTests(unittest.TestCase):
    def test_experiment_copy_excludes_dataset_trees_images_and_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('data/tiny-imagenet-200/train', 'nested/datasets',
                           'nested/tiny-imagenet-200', 'results'):
                (root / folder).mkdir(parents=True)
            for name in ('data/tiny-imagenet-200/train/image.JPEG',
                         'data/tiny-imagenet-200/train/labels.json',
                         'nested/datasets/model.pt', 'nested/tiny-imagenet-200/labels.csv',
                         'tiny-imagenet-200.zip', 'loose-image.JPEG',
                         'results/checkpoint.pt', 'results/report.json', 'results/scores.csv'):
                (root / name).write_bytes(b'test')
            client = Files()
            store = R2Store(client, 'test')
            result = store.copy_tree(root, 'uncle', exclude_dirs=('data', 'datasets', 'tiny-imagenet-200'),
                                     suffixes=('.pt', '.json', '.csv'))
            self.assertEqual(set(client.objects), {'uncle/results/checkpoint.pt',
                'uncle/results/report.json', 'uncle/results/scores.csv'})
            self.assertEqual(result['files'], 3)
            self.assertEqual(result['skipped_files'], 2)
            self.assertEqual(set(result['skipped_directories']),
                             {'data', 'nested/datasets', 'nested/tiny-imagenet-200'})

    def test_copy_retry_cache_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'drive'
            root.mkdir()
            source = root / 'model.pt'
            source.write_bytes(b'original model')
            client = Files()
            store = R2Store(client, 'test')
            self.assertEqual(store.copy_tree(root, 'uncle')['files'], 1)
            store.copy_tree(root, 'uncle')
            self.assertEqual(client.uploads, ['uncle/model.pt'])
            self.assertEqual(list(store.keys('uncle/')), ['uncle/model.pt'])
            cache = Path(directory) / 'cache'
            downloaded = store.download('uncle/model.pt', cache)
            self.assertEqual(downloaded.read_bytes(), source.read_bytes())
            store.download('uncle/model.pt', cache)
            self.assertEqual(client.downloads, 1)
            downloaded.write_bytes(b'bad cache')
            store.download('uncle/model.pt', cache)
            self.assertEqual(client.downloads, 2)
            source.write_bytes(b'different model')
            with self.assertRaises(FileExistsError):
                store.copy_tree(root, 'uncle')
            self.assertEqual(downloaded.read_bytes(), b'original model')

    def test_bad_bytes_rejected_even_with_correct_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'model.pt'
            source.write_bytes(b'original model')
            client = Files()
            client.corrupt_upload = True
            store = R2Store(client, 'test')
            with self.assertRaises(ValueError):
                store.upload(source, 'model.pt')
            with self.assertRaises(ValueError):
                store.download('model.pt', Path(directory) / 'cache')
            self.assertFalse(any((Path(directory) / 'cache').rglob('model.pt')))
            self.assertFalse(any((Path(directory) / 'cache').rglob('*.partial-*')))

    def test_completion_report_waits_for_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'settings.json').write_text('{}')
            report_path = output / 'report.json'
            report_path.write_text(json.dumps({'status': 'running'}))
            client = Files()
            store = R2Store(client, 'test')
            store.publish_lesson(output, 'lesson')
            report_path.write_text(json.dumps({'status': 'complete'}))
            with self.assertRaises(ValueError):
                store.publish_lesson(output, 'lesson')
            with self.assertRaises(FileNotFoundError):
                store.publish_lesson(output, 'lesson', complete=True)
            self.assertEqual(json.loads(client.objects['lesson/report.json'][0])['status'], 'running')
            (output / 'checkpoint.pt').write_bytes(b'model')
            store.publish_lesson(output, 'lesson', complete=True)
            self.assertLess(client.uploads.index('lesson/checkpoint.pt'), len(client.uploads) - 1)
            self.assertEqual(json.loads(client.objects['lesson/report.json'][0])['status'], 'complete')

    def test_verify_retries_a_read_that_times_out_part_way(self):
        # The upload itself succeeded; only reading it back stalled. The old code
        # let that kill the run, so check the read is simply started again.
        from botocore.exceptions import ReadTimeoutError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            path.write_bytes(b'model')
            client = Files()
            reads = []

            def flaky_get_object(Bucket, Key):
                reads.append(Key)
                if len(reads) == 1:
                    raise ReadTimeoutError(endpoint_url=None)
                return {'Body': io.BytesIO(client.objects[Key][0])}

            client.get_object = flaky_get_object
            store = R2Store(client, 'test')
            store.upload(path, 'lesson/checkpoint.pt')
            self.assertEqual(reads, ['lesson/checkpoint.pt'] * 2)


    def test_unsafe_keys_and_endpoints_rejected(self):
        store = R2Store(Files(), 'test')
        for key in ('../model', '/model', 'a/../model', 'a\\model', 'C:/model', 'a//model'):
            with self.assertRaises(ValueError):
                store.download(key, '.')
        with self.assertRaises(ValueError):
            R2Store.connect('https://untrusted.example', 'test', 'example', 'example')


if __name__ == '__main__':
    unittest.main()
