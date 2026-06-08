<?php
defined('MOODLE_INTERNAL') || die();

if ($ADMIN->fulltree) {

    $settings->add(new admin_setting_heading(
        'block_chatbot_heading',
        get_string('settings_heading', 'block_chatbot'),
        get_string('settings_heading_desc', 'block_chatbot')
    ));

    // Backend base URL (no trailing slash needed).
    $settings->add(new admin_setting_configtext(
        'block_chatbot/backend_url',
        get_string('backend_url', 'block_chatbot'),
        get_string('backend_url_desc', 'block_chatbot'),
        'https://willful-rutty-sal.ngrok-free.dev',
        PARAM_URL
    ));

    // Shared JWT secret — must match JWT_SECRET in the backend .env.
    $settings->add(new admin_setting_configpasswordunmask(
        'block_chatbot/jwt_secret',
        get_string('jwt_secret', 'block_chatbot'),
        get_string('jwt_secret_desc', 'block_chatbot'),
        'your-super-secret-jwt-key-for-local-dev'
    ));

    // JWT lifetime in seconds.
    $settings->add(new admin_setting_configtext(
        'block_chatbot/token_ttl',
        get_string('token_ttl', 'block_chatbot'),
        get_string('token_ttl_desc', 'block_chatbot'),
        '3600',
        PARAM_INT
    ));
}
