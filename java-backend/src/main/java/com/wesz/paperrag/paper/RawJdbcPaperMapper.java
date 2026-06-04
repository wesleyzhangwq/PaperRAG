package com.wesz.paperrag.paper;

import java.sql.Timestamp;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class RawJdbcPaperMapper {

    private final JdbcTemplate jdbcTemplate;

    public RawJdbcPaperMapper(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public Optional<Paper> selectById(long id) {
        return jdbcTemplate.query("""
                SELECT id, tenant_id, title, abstract_text, authors, status, doi, created_at, updated_at
                FROM papers
                WHERE id = ?
                """,
            (rs, rowNum) -> {
                Paper paper = new Paper();
                paper.setId(rs.getLong("id"));
                paper.setTenantId(rs.getLong("tenant_id"));
                paper.setTitle(rs.getString("title"));
                paper.setAbstractText(rs.getString("abstract_text"));
                paper.setAuthors(rs.getString("authors"));
                paper.setStatus(PaperStatus.valueOf(rs.getString("status")));
                paper.setDoi(rs.getString("doi"));
                paper.setCreatedAt(rs.getTimestamp("created_at").toInstant());
                Timestamp updatedAt = rs.getTimestamp("updated_at");
                paper.setUpdatedAt(updatedAt.toInstant());
                return paper;
            },
            id
        ).stream().findFirst();
    }
}
