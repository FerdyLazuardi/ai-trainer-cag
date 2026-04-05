import httpx
import asyncio
import jwt
import time
import json

async def run_test():
    # Generate JWT for Danu
    token = jwt.encode(
        {'user_id': '6', 'username': 'danu', 'role': 'moodle_user', 'exp': time.time()+3600}, 
        'your-super-secret-jwt-key-for-local-dev', 
        algorithm='HS256'
    )
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'
    }
    
    session_id = f"sess_danu_ngrok_test_{int(time.time())}"
    
    print(f"Sending chat request via ngrok (https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/api/v1/chat)...")
    async with httpx.AsyncClient(verify=False) as client:
        try:
            r = await client.post(
                'https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/api/v1/chat',
                json={
                    'query': 'Halo Peped! Ini tes E2E AFK Worker lewat Moodle ngrok yang baru ya. Saya Danu.',
                    'conversation_id': session_id,
                    'course_id': 3
                },
                headers=headers,
                timeout=60.0
            )
            print(f"Chat API status: {r.status_code}")
            print(f"Chat API response: {r.text[:300]}")
        except Exception as e:
            print(f"Error hitting ngrok Chat API: {e}")
            return
            
        print("\nWaiting 15 seconds for AFK Worker to process the memory...")
        await asyncio.sleep(15)
        
        print("\nChecking Qdrant at Proxmox IP: http://172.16.10.235:6335...")
        try:
            q_payload = {
                "filter": {
                    "must": [
                        {"key": "session_id", "match": {"value": session_id}}
                    ]
                },
                "with_payload": True
            }
            q_r = await client.post('http://172.16.10.235:6335/collections/user_ltm_memories/points/scroll', json=q_payload)
            print(f"Qdrant status: {q_r.status_code}")
            points = q_r.json().get('result', {}).get('points', [])
            print(f"Found {len(points)} memory points for session {session_id} in LTM.")
            if points:
                print("\nLTM Payload Stored:")
                print(json.dumps(points[0]['payload'], indent=2))
            else:
                print("No points found. The AFK worker might need more time or something failed.")
        except Exception as e:
            print(f"Error querying Qdrant: {e}")

asyncio.run(run_test())
