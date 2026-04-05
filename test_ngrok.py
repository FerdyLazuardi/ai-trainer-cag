import httpx
import asyncio
import re
import json
import time

async def test_e2e():
    async with httpx.AsyncClient(verify=False) as client:
        # 1. Get login token
        r1 = await client.get('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/login/index.php', headers={'ngrok-skip-browser-warning': 'true'})
        match = re.search(r'name=\"logintoken\" value=\"([a-zA-Z0-9_]+)\"', r1.text)
        logintoken = match.group(1) if match else ''
        print(f"Logintoken: {logintoken}")
        
        # 2. Login
        r2 = await client.post('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/login/index.php', 
            data={'username': 'danu', 'password': '@Danu123', 'logintoken': logintoken},
            headers={'ngrok-skip-browser-warning': 'true'},
            follow_redirects=True
        )
        
        # 3. Access dashboard to extract tokens and constants
        r3 = await client.get('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/my/', headers={'ngrok-skip-browser-warning': 'true'})
        
        jwt_match = re.search(r'const MOODLE_JWT = \"(.*?)\"', r3.text)
        api_base_match = re.search(r'const API_BASE_URL = \"(.*?)\"', r3.text)
        user_id_match = re.search(r'const MOODLE_USER_ID = (.*?);', r3.text)
        user_name_match = re.search(r'const MOODLE_USER_NAME = \"(.*?)\"', r3.text)
        dept_match = re.search(r'const MOODLE_DEPT = \"(.*?)\"', r3.text)
        
        jwt_token = jwt_match.group(1) if jwt_match else None
        api_base = api_base_match.group(1) if api_base_match else 'https://semiexpositive-renaldo-unvindictively.ngrok-free.dev'
        user_id = user_id_match.group(1) if user_id_match else '0'
        user_name = user_name_match.group(1) if user_name_match else 'user'
        dept = dept_match.group(1) if dept_match else 'general'
        
        print(f"Extracted JWT: {'Found' if jwt_token else 'Not found'}")
        print(f"API Base: {api_base}")
        print(f"User ID: {user_id}")
        
        if not jwt_token:
            print("Cannot proceed without JWT. Finding all constants:")
            print(re.findall(r'const MOODLE_.*', r3.text))
            return
            
        # Build session ID exactly as script.js does
        clean_name = user_name.replace(' ', '_').lower()
        clean_dept = dept.lower()
        session_id = f"{clean_name}_{user_id}_{clean_dept}"
        print(f"Session ID: {session_id}")
        
        # 4. Send chat message
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jwt_token}',
            'ngrok-skip-browser-warning': 'true'
        }
        
        # Adding a unique string to easily find it in Qdrant
        unique_msg = f"Halo, saya Danu menguji langsung dari ngrok Moodle pada {time.time()}"
        
        payload = {
            'query': unique_msg,
            'conversation_id': session_id,
            'course_id': 3,
            'course_name': 'AI_Knowledge_Base'
        }
        
        print(f"Sending message: {unique_msg}")
        r4 = await client.post(f"{api_base}/api/v1/chat", json=payload, headers=headers, timeout=60.0)
        print(f"Chat status: {r4.status_code}")
        print(f"Chat response: {r4.text[:200]}")
        
        # 5. Wait for AFK worker to trigger (15 seconds)
        print("Waiting 15 seconds for AFK worker...")
        await asyncio.sleep(15)
        
        # 6. Check Qdrant directly on the proxmox IP
        print("Checking Qdrant...")
        qdrant_url = "http://172.16.10.235:6335/collections/user_ltm_memories/points/scroll"
        qdrant_payload = {
            "filter": {
                "must": [
                    {"key": "session_id", "match": {"value": session_id}}
                ]
            },
            "with_payload": True
        }
        r5 = await client.post(qdrant_url, json=qdrant_payload)
        print(f"Qdrant status: {r5.status_code}")
        
        data = r5.json()
        points = data.get('result', {}).get('points', [])
        print(f"Found {len(points)} points for session {session_id}")
        if points:
            print(json.dumps(points[0]['payload'], indent=2))

asyncio.run(test_e2e())
