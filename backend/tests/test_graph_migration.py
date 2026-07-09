from unittest.mock import MagicMock, patch

from app.db import mysql


@patch("app.db.mysql.inspect")
def test_migrate_papers_adds_only_missing_graph_columns(mock_inspect) -> None:
    inspector = MagicMock()
    inspector.get_table_names.return_value = ["papers"]
    inspector.get_columns.return_value = [{"name": "paper_id"}]
    mock_inspect.return_value = inspector
    connection = MagicMock()

    with patch.object(mysql.engine, "begin") as begin:
        begin.return_value.__enter__.return_value = connection
        mysql._migrate_papers()

    statements = "\n".join(str(call.args[0]) for call in connection.execute.call_args_list)
    assert "graph_sync_status" in statements
    assert "graph_synced_at" in statements
    assert "graph_sync_error" in statements
    assert "ix_papers_graph_sync_status" in statements
