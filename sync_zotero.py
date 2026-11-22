"""Push library.ris into a Zotero collection. Adds only, never edits or deletes.

    python bibliography/sync_zotero.py            # dry run, shows what it would add
    python bibliography/sync_zotero.py --apply    # actually writes

Reads ZOTERO_USER_ID and ZOTERO_API_KEY from .env at the project root.
Skips anything already in your library, matched on DOI first, then on title.

It only ever sends GET and POST-new-item requests. There is no code path here that
can modify or remove an item you already have.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIS = os.path.join(ROOT, "bibliography", "library.ris")
COLLECTION = "Continual learning: forgetting"
API = "https://api.zotero.org"


# ---------- credentials ----------

def load_env():
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        sys.exit("No .env at the project root. Copy .env.example and fill it in.")
    env = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("\"'")
    uid, key = env.get("ZOTERO_USER_ID"), env.get("ZOTERO_API_KEY")
    if not uid or not key:
        sys.exit("Need ZOTERO_USER_ID and ZOTERO_API_KEY in .env")
    if not uid.isdigit():
        sys.exit("ZOTERO_USER_ID should be all digits. Check https://www.zotero.org/settings/keys")
    return uid, key


# ---------- http ----------

def call(method, path, key, body=None, params=None):
    if method not in ("GET", "POST"):
        raise RuntimeError(f"refusing {method}: this script is add-only")
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Zotero-API-Key", key)
    req.add_header("Zotero-API-Version", "3")
    if data:
        req.add_header("Content-Type", "application/json")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                wait = resp.headers.get("Backoff")
                if wait:
                    time.sleep(float(wait))
                return json.loads(resp.read() or "null"), dict(resp.headers)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(float(e.headers.get("Retry-After", 2 ** attempt)))
                continue
            hint = {
                403: "key rejected, or it lacks library access. Check the key's permissions.",
                404: "user ID not found. Check ZOTERO_USER_ID.",
            }.get(e.code, e.read().decode(errors="replace")[:300])
            sys.exit(f"Zotero returned {e.code}: {hint}")
    sys.exit("Zotero kept rate-limiting or erroring. Try again later.")


# ---------- ris ----------

TYPE_MAP = {"CONF": "conferencePaper", "RPRT": "preprint", "JOUR": "journalArticle"}


def parse_ris(path):
    entries, cur = [], {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line.rstrip("\n"))
            if not m:
                continue
            tag, val = m.group(1), m.group(2).strip()
            if tag == "TY":
                cur = {"TY": val, "AU": [], "KW": []}
            elif tag == "ER":
                if cur:
                    entries.append(cur)
                cur = {}
            elif tag in ("AU", "KW"):
                cur.setdefault(tag, []).append(val)
            elif cur:
                cur[tag] = val
    return entries


def to_zotero(e, collection_key):
    itype = TYPE_MAP.get(e["TY"], "document")
    creators = []
    for name in e.get("AU", []):
        last, _, first = name.partition(",")
        creators.append({
            "creatorType": "author",
            "lastName": last.strip(),
            "firstName": first.strip(),
        })
    item = {
        "itemType": itype,
        "title": e.get("TI", ""),
        "creators": creators,
        "abstractNote": e.get("AB", ""),
        "url": e.get("UR", ""),
        "DOI": e.get("DO", ""),
        "tags": [{"tag": t} for t in e.get("KW", [])],
        "collections": [collection_key],
    }
    if itype == "conferencePaper":
        item["proceedingsTitle"] = e.get("T2", "")
        item["date"] = e.get("PY", "")
        if e.get("SP") and e.get("EP"):
            item["pages"] = f"{e['SP']}-{e['EP']}"
    elif itype == "preprint":
        item["repository"] = "arXiv"
        item["date"] = e.get("PY", "")
        m = re.search(r"arXiv:(\S+)", e.get("T2", ""))
        if m:
            item["archiveID"] = f"arXiv:{m.group(1)}"
        item.pop("DOI", None)  # preprint items carry the id in archiveID instead
    else:
        item["date"] = e.get("PY", "")
    return item


def norm(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


# ---------- library state ----------

def existing(uid, key):
    """DOIs and normalised titles already in the library."""
    dois, titles, start = set(), set(), 0
    while True:
        page, _ = call("GET", f"/users/{uid}/items/top", key,
                       params={"format": "json", "limit": 100, "start": start})
        if not page:
            break
        for it in page:
            d = it.get("data", {})
            if d.get("DOI"):
                dois.add(d["DOI"].lower().replace("https://doi.org/", ""))
            if d.get("title"):
                titles.add(norm(d["title"]))
        if len(page) < 100:
            break
        start += 100
    return dois, titles


def find_or_create_collection(uid, key, name, apply):
    start = 0
    while True:
        page, _ = call("GET", f"/users/{uid}/collections", key,
                       params={"format": "json", "limit": 100, "start": start})
        if not page:
            break
        for c in page:
            if c["data"]["name"] == name:
                return c["data"]["key"], False
        if len(page) < 100:
            break
        start += 100
    if not apply:
        return "DRYRUN0000", True
    res, _ = call("POST", f"/users/{uid}/collections", key, body=[{"name": name}])
    created = res.get("successful", {})
    if not created:
        sys.exit(f"Could not create the collection: {res.get('failed')}")
    return created["0"]["key"], True


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write to Zotero")
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()

    uid, key = load_env()
    entries = parse_ris(RIS)
    print(f"Read {len(entries)} entries from library.ris")

    print("Checking what you already have...")
    dois, titles = existing(uid, key)
    print(f"Your library has {len(titles)} top-level items")

    ckey, is_new = find_or_create_collection(uid, key, args.collection, args.apply)
    print(f"Collection {args.collection!r}: {'will be created' if is_new else 'exists'}")

    todo, skip = [], []
    for e in entries:
        doi = e.get("DO", "").lower()
        if (doi and doi in dois) or norm(e.get("TI", "")) in titles:
            skip.append(e)
        else:
            todo.append(e)

    for e in skip:
        print(f"  skip   {e.get('TI','')[:68]}")
    for e in todo:
        print(f"  add    {e.get('TI','')[:68]}")

    if not todo:
        print("\nNothing to add. Your library is up to date.")
        return

    if not args.apply:
        print(f"\nDry run. {len(todo)} item(s) would be added, nothing written.")
        print("Run again with --apply to write them.")
        return

    items = [to_zotero(e, ckey) for e in todo]
    res, _ = call("POST", f"/users/{uid}/items", key, body=items)
    ok, failed = res.get("successful", {}), res.get("failed", {})
    print(f"\nAdded {len(ok)} item(s)" + (f", {len(failed)} failed" if failed else ""))
    for idx, err in failed.items():
        print(f"  failed: {todo[int(idx)].get('TI','')[:50]} -> {err.get('message')}")

    # reliability warnings ride along as child notes
    notes = []
    for idx, created in ok.items():
        n1 = todo[int(idx)].get("N1")
        if n1:
            notes.append({"itemType": "note", "parentItem": created["key"],
                          "note": f"<p>{n1}</p>"})
    if notes:
        call("POST", f"/users/{uid}/items", key, body=notes)
        print(f"Attached {len(notes)} note(s)")

    print("\nPDFs are in papers/ and were not uploaded. Attach them in Zotero with "
          "right-click > Add Attachment > Attach Link to File.")


if __name__ == "__main__":
    main()
