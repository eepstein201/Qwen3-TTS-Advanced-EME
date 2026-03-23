module.exports = {
  apps: [
    {
      name: 'tts-server-5123',
      cwd: '/Users/ericepstein/Qwen3-TTS_UserFiles',
      script: 'start.cjs',
      interpreter: '/opt/homebrew/bin/node',
      env: {
        PYTHONUNBUFFERED: '1',
      },
      watch: false,
      autorestart: true,
      max_restarts: 3,
      restart_delay: 2000,
    },
  ],
};
