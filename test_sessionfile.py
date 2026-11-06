"""
A session goes where the user said, and comes back.

The store by name is gone: these are the two things that replaced it, and the
cases worth asserting are the ones where a path is not quite what the dialog
hands over -- a name with no suffix, a name that already has it, a folder that
does not exist yet, and a file that is somebody else's JSON.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import sessionfile as SF


def test_saves_where_told_and_adds_the_suffix_once():
    with TemporaryDirectory() as d:
        a = SF.save(Path(d) / "aorta 14", {"tab": "wall"})
        assert a.name == "aorta 14.asfq.json"
        b = SF.save(Path(d) / "aorta 14.asfq.json", {"tab": "wall"})
        assert b == a                      # not "aorta 14.asfq.json.asfq.json"
        assert SF.load(a)["ui"]["tab"] == "wall"
        assert SF.load(a)["name"] == "aorta 14"


def test_makes_the_folder_the_dialog_just_named():
    with TemporaryDirectory() as d:
        p = SF.save(Path(d) / "new folder" / "s", {})
        assert p.is_file()


def test_reads_a_session_written_under_the_old_app_name():
    with TemporaryDirectory() as d:
        p = Path(d) / "old.asfq.json"
        p.write_text('{"app": "fibrosis-quantifier", "ui": {"tab": "aorta"}}')
        assert SF.load(p)["ui"]["tab"] == "aorta"


def test_refuses_json_that_is_not_a_session():
    with TemporaryDirectory() as d:
        p = Path(d) / "layout.json"
        p.write_text('{"poses": []}')
        with pytest.raises(ValueError):
            SF.load(p)
        with pytest.raises(FileNotFoundError):
            SF.load(Path(d) / "nothing.asfq.json")
