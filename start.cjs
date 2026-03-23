const { spawn } = require('child_process');

// Activate conda env and run the FastAPI server
const proc = spawn(
  '/bin/zsh',
  [
    '-c',
    'source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts-mlx && python -m qwen3_tts.server.app',
  ],
  {
    cwd: __dirname,
    stdio: 'inherit',
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  }
);

proc.on('close', (code) => process.exit(code || 0));
