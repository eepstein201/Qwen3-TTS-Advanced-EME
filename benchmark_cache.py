import asyncio
import base64
import os
import tempfile
import time

# We will simulate the exact synchronous / asynchronous operations
# that happen in app_generation.py without heavy dependencies

async def measure_event_loop_delay(duration=5.0):
    """Measures the max event loop delay over the duration."""
    max_delay = 0.0
    start = time.time()

    while time.time() - start < duration:
        t0 = time.time()
        await asyncio.sleep(0.001)
        delay = (time.time() - t0) - 0.001
        if delay > max_delay:
            max_delay = delay

    return max_delay

def sync_read_b64(cache_file):
    """The current unoptimized way"""
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

async def async_read_b64(cache_file):
    """The optimized way using asyncio.to_thread"""
    if cache_file and os.path.exists(cache_file):
        def _read():
            with open(cache_file, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return await asyncio.to_thread(_read)
    return None

async def simulate_endpoint_hit(cache_files, use_async, runs=20):
    for _ in range(runs):
        if use_async:
            results = await asyncio.gather(*[async_read_b64(f) for f in cache_files])
        else:
            results = [sync_read_b64(f) for f in cache_files]
        await asyncio.sleep(0.01) # simulated other work

async def run_benchmark(files, use_async):
    loop_task = asyncio.create_task(measure_event_loop_delay(3.0))
    # simulate 10 concurrent requests, each asking for 10 files
    work_tasks = [asyncio.create_task(simulate_endpoint_hit(files, use_async)) for _ in range(10)]

    await asyncio.gather(*work_tasks)
    max_delay = await loop_task
    return max_delay

async def main():
    print("Setting up test cache files...")
    # Create some large dummy wav files to simulate long texts/heavy I/O
    files = []

    # Generate 10MB of random bytes per file
    dummy_data = os.urandom(10 * 1024 * 1024)

    for _ in range(10):
        fd, path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, 'wb') as f:
            f.write(dummy_data)
        files.append(path)

    try:
        print("Running benchmark: Synchronous (Baseline)")
        sync_delay = await run_benchmark(files, use_async=False)
        print(f"Max Event Loop Delay (Sync): {sync_delay*1000:.2f} ms")

        print("\nRunning benchmark: Asynchronous (Optimized)")
        async_delay = await run_benchmark(files, use_async=True)
        print(f"Max Event Loop Delay (Async): {async_delay*1000:.2f} ms")

    finally:
        for f in files:
            try:
                os.remove(f)
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main())
