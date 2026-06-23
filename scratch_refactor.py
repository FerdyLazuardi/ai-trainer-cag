with open('app/api/routes/chat.py', 'r') as f:
    lines = f.readlines()

cache_block = lines[532:556]  # 533 to 556 (0-indexed 532 to 555)

# remove from original place
del lines[532:556]

# find new place (after gather)
insert_idx = 0
for i, line in enumerate(lines):
    if 'logger.debug(f"[TIMING] gather(history+ltm+profile):' in line:
        insert_idx = i + 1
        break

new_block = [
    '    if recent_history or summary:\n',
    '        skip_cache = True\n',
    '        logger.debug("Cache lookup skipped - conversation history is not empty, query is context-dependent")\n\n'
] + cache_block

lines = lines[:insert_idx] + new_block + lines[insert_idx:]

with open('app/api/routes/chat.py', 'w') as f:
    f.writelines(lines)
