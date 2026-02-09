module.exports = {
  apps: [
    {
      name: "rag-backend",
      cwd: "./backend",
      script: "/home/ragssh/.venv/bin/python",
      args: "main.py",
      interpreter: "none",
      env: {
        PORT: 8000,
        PYTHONUNBUFFERED: "1",
        HF_HUB_OFFLINE: "1",
        TOKENIZERS_PARALLELISM: "false"
      }
    },
    {
      name: "rag-frontend",
      cwd: "./frontend",
      script: "npm",
      args: "start",
      env: {
        PORT: 3000,
        NODE_ENV: "production"
      }
    }
  ]
};
