import httpx
import asyncio
import re

async def get_token():
    async with httpx.AsyncClient(verify=False) as client:
        r1 = await client.get('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/login/index.php', headers={'ngrok-skip-browser-warning': 'true'})
        # Find logintoken
        match = re.search(r'name=\"logintoken\" value=\"([a-zA-Z0-9_]+)\"', r1.text)
        logintoken = match.group(1) if match else ''
        
        r2 = await client.post('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/login/index.php', 
            data={'username': 'danu', 'password': '@Danu123', 'logintoken': logintoken},
            headers={'ngrok-skip-browser-warning': 'true'}
        )
        
        r3 = await client.get('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/my/', headers={'ngrok-skip-browser-warning': 'true'})
        
        jwt_match = re.search(r'const MOODLE_JWT = \"(.*?)\"', r3.text)
        if jwt_match:
            print("JWT:", jwt_match.group(1))
        else:
            print("Login success:", "Dashboard" in r3.text or "Log out" in r3.text)
            print("Tokens found in text:", re.findall(r'MOODLE_[A-Z]+ = .*', r3.text))

asyncio.run(get_token())
