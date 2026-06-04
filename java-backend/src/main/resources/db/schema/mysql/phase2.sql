CREATE TABLE IF NOT EXISTS papers (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id     BIGINT       NOT NULL,
  title         VARCHAR(512) NOT NULL,
  abstract_text TEXT,
  authors       VARCHAR(512),
  status        VARCHAR(32)  NOT NULL DEFAULT 'PENDING',
  doi           VARCHAR(128),
  created_at    DATETIME     NOT NULL,
  updated_at    DATETIME     NOT NULL,
  KEY idx_tenant_status_time (tenant_id, status, created_at),
  UNIQUE KEY uk_tenant_doi (tenant_id, doi)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ingest_tasks (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  paper_id      BIGINT       NOT NULL,
  tenant_id     BIGINT       NOT NULL,
  biz_key       VARCHAR(128) NOT NULL,
  status        VARCHAR(32)  NOT NULL,
  error_message VARCHAR(1024),
  created_at    DATETIME     NOT NULL,
  updated_at    DATETIME     NOT NULL,
  CONSTRAINT fk_ingest_tasks_paper FOREIGN KEY (paper_id) REFERENCES papers(id),
  UNIQUE KEY uk_ingest_task_tenant_biz_key (tenant_id, biz_key),
  UNIQUE KEY uk_ingest_task_paper_biz_key (paper_id, biz_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS paper_chunks (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  paper_id    BIGINT NOT NULL,
  tenant_id   BIGINT NOT NULL,
  chunk_index INT    NOT NULL,
  content     MEDIUMTEXT NOT NULL,
  token_count INT    NOT NULL,
  created_at  DATETIME NOT NULL,
  CONSTRAINT fk_paper_chunks_paper FOREIGN KEY (paper_id) REFERENCES papers(id),
  UNIQUE KEY uk_paper_chunk_index (paper_id, chunk_index),
  KEY idx_paper_chunks_tenant_paper (tenant_id, paper_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
