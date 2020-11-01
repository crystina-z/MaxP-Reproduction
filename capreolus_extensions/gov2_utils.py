import os
from time import time
import gzip
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

DOC, TERMINATING_DOC = "<DOC>", "</DOC>"
DOCNO, TERMINATING_DOCNO = "<DOCNO>", "</DOCNO>"
DOCHDR, TERMINATING_DOCHDR = "<DOCHDR>", "</DOCHDR>"


def clean_documents(docs):
    docs = docs.replace(f"{DOC}\n", "").replace(f"{TERMINATING_DOC}\n", "")
    return [" ".join(d.split()).strip() for d in docs.split("\n")]


def get_next_record_pos(f):
    start_pos = f.tell()
    found = False
    line, doc_lines = f.readline(), []
    line = line.decode("utf-8")
    while line != "":
        if line.startswith(DOC):
            found = True

        if found:
            doc_lines.append(line)

        if line.startswith(TERMINATING_DOC) and len(doc_lines) != 0:
            doc_lines = clean_documents("".join(doc_lines))
            assert doc_lines[0].startswith(DOCNO)
            id = parse_record(doc_lines)
            return id, (start_pos, f.tell() - start_pos)

        line = f.readline().decode("utf-8", errors="ignore")

    return "END", (-1, -1)


def parse_record(doc_lines, return_doc=False):
    """ according to: https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/collection/TrecwebCollection.java#L78-L98 """

    if isinstance(doc_lines, list):
        doc = " ".join(doc_lines)
    else:
        doc = doc_lines

    i = doc.index(DOCNO)
    if i == -1:
        raise ValueError("cannot find start tag " + DOCNO)
    if i != 0:
        raise ValueError("should start with " + DOCNO)

    j = doc.index(TERMINATING_DOCNO)
    if j == -1:
        raise ValueError("cannot find end tag " + TERMINATING_DOCNO)

    id = doc[i+len(DOCNO):j].replace(DOCNO, "").replace(TERMINATING_DOCNO, "")

    if not return_doc:
        return id

    i = doc.index(DOCHDR)
    if i == -1:
        raise ValueError("cannot find header tag " + DOCHDR)

    j = doc.index(TERMINATING_DOCHDR)
    if j == -1:
        raise ValueError("cannot find end tag " + TERMINATING_DOCHDR)
    if j < i:
        raise ValueError(TERMINATING_DOCHDR + " comes before " + DOCHDR)

    doc = doc[j+len(TERMINATING_DOCHDR):].replace(TERMINATING_DOCHDR, "").replace(TERMINATING_DOC, "").strip()
    return id, doc


def spawn_child_process_to_read_docs(data):
    path, shared_id2pos = data["path"], data["shared_id2pos"]

    start = time()
    local_id2pos = {}
    for gz_fn in os.listdir(path):
        with gzip.open(f"{path}/{gz_fn}") as f:
            id, pos = get_next_record_pos(f)
            while id != "END":
                local_id2pos[id] = pos
                id, pos = get_next_record_pos(f)

    shared_id2pos.update(local_id2pos)
    logger.info("PID: {0}, Done getting documents from disk: {1} for path: {2}".format(os.getpid(), time() - start, path))

