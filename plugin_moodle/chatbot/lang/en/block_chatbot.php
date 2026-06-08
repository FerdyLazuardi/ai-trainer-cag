<?php
$string['pluginname']            = 'AI Chatbot';
$string['chatbot:addinstance']   = 'Add AI Chatbot block';
$string['chatbot:myaddinstance'] = 'Add AI Chatbot block to my page';

// Settings page.
$string['settings_heading']      = 'Ava Chatbot backend';
$string['settings_heading_desc'] = 'Configure the connection between this Moodle block and the Ava RAG backend.';
$string['backend_url']           = 'Backend URL';
$string['backend_url_desc']      = 'Base URL of the Ava backend API (no trailing slash). Example: https://your-backend.example.com';
$string['jwt_secret']            = 'JWT shared secret';
$string['jwt_secret_desc']       = 'Shared secret used to sign the JWT. Must match JWT_SECRET in the backend .env exactly (min 32 chars).';
$string['token_ttl']             = 'Token lifetime (seconds)';
$string['token_ttl_desc']        = 'How long each generated JWT stays valid. Default 3600 (1 hour).';
