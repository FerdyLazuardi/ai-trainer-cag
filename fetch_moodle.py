import httpx
import re
import asyncio

async def fetch_moodle_dashboard():
    # Use a persistent client to maintain cookies
    async with httpx.AsyncClient(verify=False, follow_redirects=True) as client:
        # 1. Get the login page to extract the logintoken
        print("Getting login page...")
        r_get = await client.get('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/login/index.php', headers={'ngrok-skip-browser-warning': 'true'})
        
        match = re.search(r'name=\"logintoken\" value=\"(.*?)\"', r_get.text)
        logintoken = match.group(1) if match else None
        print(f"Extracted logintoken: {logintoken}")
        
        if not logintoken:
            print("Failed to find logintoken.")
            return

        # 2. Perform the login POST request
        print("Logging in as 'danu'...")
        payload = {
            'username': 'danu',
            'password': '@Danu123',
            'logintoken': logintoken
        }
        r_post = await client.post('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/login/index.php', data=payload, headers={'ngrok-skip-browser-warning': 'true'})
        
        # 3. Access the dashboard
        print("Accessing dashboard (/my/)...")
        r_dash = await client.get('https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/my/', headers={'ngrok-skip-browser-warning': 'true'})
        
        with open('moodle_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(r_dash.text)
            
        print(f"Dashboard title: {re.search(r'<title>(.*?)</title>', r_dash.text).group(1)}")
        
        jwt_match = re.search(r'MOODLE_JWT\s*=\s*[\"\'\](.*?)[\"\'\]', r_dash.text)
        api_match = re.search(r'API_BASE_URL\s*=\s*[\"\'\](.*?)[\"\'\]', r_dash.text)
        user_id_match = re.search(r'MOODLE_USER_ID\s*=\s*(.*?);', r_dash.text)
        
        print(f"MOODLE_JWT found: {'Yes' if jwt_match else 'No'}")
        if jwt_match:
            print(f"JWT: {jwt_match.group(1)[:20]}...")
        if api_match:
            print(f"API_BASE_URL: {api_match.group(1)}")
        if user_id_match:
            print(f"USER_ID: {user_id_match.group(1)}")

asyncio.run(fetch_moodle_dashboard())
