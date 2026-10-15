-- Example relational schema. Adjust types for your database engine.
CREATE TABLE job_postings (
    job_id BIGINT PRIMARY KEY,
    title VARCHAR(500),
    link TEXT,
    published_date DATETIME,
    is_hourly BOOLEAN,
    hourly_low DECIMAL(12,2),
    hourly_high DECIMAL(12,2),
    budget DECIMAL(12,2),
    country VARCHAR(200),
    job_category VARCHAR(100)
);

-- Example queries
-- SELECT country, COUNT(*) AS postings
-- FROM job_postings GROUP BY country ORDER BY postings DESC;
-- SELECT job_category, COUNT(*) AS postings
-- FROM job_postings GROUP BY job_category ORDER BY postings DESC;
