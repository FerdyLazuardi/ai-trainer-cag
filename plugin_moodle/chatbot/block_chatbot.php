<?php
defined('MOODLE_INTERNAL') || die();

/**
 * AI Chatbot (Ava) — floating chat widget block.
 *
 * Injects the Ava chat widget on Moodle pages and mints a short-lived
 * HS256 JWT from the logged-in Moodle user. The JWT is read by the
 * backend (app/api/auth.py) which requires the claims: user_id, exp, iat.
 */
class block_chatbot extends block_base {

    public function init() {
        $this->title = get_string('pluginname', 'block_chatbot');
    }

    public function applicable_formats() {
        return ['all' => true];
    }

    public function instance_allow_multiple() {
        return false;
    }

    public function hide_header() {
        return true;
    }

    public function instance_can_be_docked() {
        return false;
    }

    public function has_config() {
        return true;
    }

    /**
     * Generate a JWT token (HS256) — pure PHP, no library needed.
     */
    private function generate_jwt(array $payload, string $secret): string {
        $b64 = fn($d) => rtrim(strtr(base64_encode($d), '+/', '-_'), '=');
        $header      = $b64(json_encode(['alg' => 'HS256', 'typ' => 'JWT']));
        $payload_enc = $b64(json_encode($payload));
        $sig = $b64(hash_hmac('sha256', "$header.$payload_enc", $secret, true));
        return "$header.$payload_enc.$sig";
    }

    public function get_content() {
        global $USER, $CFG, $COURSE;

        if ($this->content !== null) {
            return $this->content;
        }

        $this->content = new stdClass();

        // Don't render for logged-out / guest sessions — no user to authenticate.
        if (!isloggedin() || isguestuser()) {
            $this->content->text = '';
            $this->content->footer = '';
            return $this->content;
        }

        // Read admin-configured values, falling back to sane dev defaults.
        $BASE_URL   = get_config('block_chatbot', 'backend_url');
        $JWT_SECRET = get_config('block_chatbot', 'jwt_secret');

        if (empty($BASE_URL)) {
            $BASE_URL = 'https://willful-rutty-sal.ngrok-free.dev';
        }
        if (empty($JWT_SECRET)) {
            // ⚠️ Harus sama persis dengan JWT_SECRET di file .env backend.
            $JWT_SECRET = 'your-super-secret-jwt-key-for-local-dev';
        }
        $BASE_URL = rtrim($BASE_URL, '/');

        $token_ttl = (int)get_config('block_chatbot', 'token_ttl');
        if ($token_ttl <= 0) {
            $token_ttl = 3600;
        }

        $user_id     = (int)$USER->id;
        $user_name   = addslashes(fullname($USER));
        $user_dept   = addslashes($USER->department ?? '');
        $course_id   = (int)$COURSE->id;
        $course_name = addslashes($COURSE->fullname ?? 'Dashboard');
        $sesskey     = sesskey();

        // Format user_id agar sama dengan conversation_id di JS: {nama}_{id}_{dept}
        // Contoh: "admin_2_general", "danu_lintang_9_fo"
        $name_slug = strtolower(preg_replace('/\s+/', '_', fullname($USER)));
        $dept_slug = strtolower($USER->department ?? 'general') ?: 'general';
        $user_id_readable = "{$name_slug}_{$user_id}_{$dept_slug}";

        // Buat JWT dari data user Moodle — dibaca oleh auth.py di backend.
        $jwt = $this->generate_jwt([
            'user_id'    => $user_id,
            'session_id' => $user_id_readable,   // e.g. "admin_2_general"
            'username'   => fullname($USER),
            'role'       => 'moodle_user',
            'dept'       => $USER->department ?? '',
            'iat'        => time(),
            'exp'        => time() + $token_ttl,
        ], $JWT_SECRET);

        $this->content->text = <<<HTML
<style>
    .block_chatbot,
    .block_chatbot .card,
    .block_chatbot .card-body {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        min-height: 0 !important;
        height: 0 !important;
    }
</style>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
<link rel="stylesheet" href="{$CFG->wwwroot}/blocks/chatbot/static/style.css">

<script>
    var MOODLE_USER_ID     = {$user_id};
    var MOODLE_USER_NAME   = "{$user_name}";
    var MOODLE_DEPT        = "{$user_dept}";
    var MOODLE_COURSE_ID   = {$course_id};
    var MOODLE_COURSE_NAME = "{$course_name}";
    var MOODLE_SESSKEY     = "{$sesskey}";
    var MOODLE_JWT         = "{$jwt}";
    var API_BASE_URL       = "{$BASE_URL}";
</script>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="{$CFG->wwwroot}/blocks/chatbot/static/script.js"></script>
HTML;

        $this->content->footer = '';
        return $this->content;
    }
}
