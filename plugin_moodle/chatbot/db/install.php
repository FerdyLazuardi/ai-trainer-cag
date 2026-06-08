<?php
defined('MOODLE_INTERNAL') || die();

/**
 * Auto-place the chatbot block site-wide on first install so it appears
 * on every page without an admin having to add it manually.
 */
function xmldb_block_chatbot_install() {
    global $DB;

    if ($DB->record_exists('block_instances', ['blockname' => 'chatbot'])) {
        return true;
    }

    $syscontext = context_system::instance();

    $DB->insert_record('block_instances', (object)[
        'blockname'         => 'chatbot',
        'parentcontextid'   => $syscontext->id,
        'showinsubcontexts' => 1,
        'requiredbytheme'   => 0,
        'pagetypepattern'   => '*',
        'subpagepattern'    => null,
        'defaultregion'     => 'side-pre',
        'defaultweight'     => -10,
        'timecreated'       => time(),
        'timemodified'      => time(),
        'configdata'        => ''
    ]);

    return true;
}
