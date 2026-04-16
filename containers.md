### Redis for chat history
docker run --restart always --name hyper-redis -d -p 6380:6379 redis redis-server --save 60 1

### Langfuse for Tracing and Evaluation
docker compose up (in langfuse folder)

uses ports: 
- 5432 (Postgres)
- 3000 (langfuse-web)
- 3030 (langfuse-worker)
- 8123 (clickhouse)
- 9000:9000 (clickhouse)
- 9090:9000 (minio)
- 9091:9001 (minio)
- 6379:6379 (redis)

### LiteLLM
docker compose up (in ./liteLLM folder)

uses ports:
- 4000 (litellm)
- 5435:5432 (Postgres)
- 9092:9090 (prometheus)
