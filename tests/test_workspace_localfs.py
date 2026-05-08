"""Workspace LocalFsAdapter · write/read/list/delete + 路径逃逸防御"""

from __future__ import annotations

import pytest

from akong_agent_harness import LocalFsAdapter, Workspace, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    return LocalFsAdapter("ag_test", root=tmp_path)


def test_write_then_read_roundtrip(ws):
    ws.write_file("notes/today.md", "hello world")
    assert ws.read_file("notes/today.md") == "hello world"


def test_list_dir_after_writes(ws):
    ws.write_file("a.md", "1")
    ws.write_file("b.md", "2")
    ws.write_file("sub/c.md", "3")
    top = ws.list_dir("")
    assert "a.md" in top
    assert "b.md" in top
    assert "sub" in top
    assert ws.list_dir("sub") == ["c.md"]


def test_delete_file_and_dir(ws):
    ws.write_file("x.md", "x")
    ws.write_file("dir/y.md", "y")
    ws.delete("x.md")
    ws.delete("dir")
    assert "x.md" not in ws.list_dir("")
    assert "dir" not in ws.list_dir("")


def test_path_escape_blocked(ws):
    with pytest.raises(WorkspaceError):
        ws.write_file("../escape.md", "no")


def test_connect_factory_returns_protocol_compliant(tmp_path, monkeypatch):
    monkeypatch.setenv("AKONG_WORKSPACE_ROOT", str(tmp_path))
    inst = Workspace.connect("ag_factory")  # type: ignore[attr-defined]
    inst.write_file("k.md", "v")
    assert inst.read_file("k.md") == "v"
