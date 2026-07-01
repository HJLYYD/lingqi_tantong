#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CONFIG_KEYS     64
#define MAX_KEY_LEN         64
#define MAX_VALUE_LEN       256
#define MAX_SECTION_LEN     64
#define MAX_CONFIG_DEPTH    16

typedef enum {
    CONFIG_TYPE_INT,
    CONFIG_TYPE_FLOAT,
    CONFIG_TYPE_BOOL,
    CONFIG_TYPE_STRING,
    CONFIG_TYPE_FLOAT_ARRAY
} ConfigValueType;

typedef struct {
    char key[MAX_KEY_LEN];
    ConfigValueType type;
    union {
        int int_val;
        float float_val;
        bool bool_val;
        char string_val[MAX_VALUE_LEN];
        struct { float data[8]; int len; } float_array;
    } value;
} ConfigEntry;

typedef struct {
    char section[MAX_SECTION_LEN];
    ConfigEntry entries[MAX_CONFIG_KEYS];
    int num_entries;
} ConfigSection;

typedef struct {
    ConfigSection sections[MAX_CONFIG_DEPTH];
    int num_sections;
    char config_path[MAX_PATH_LEN];
} ConfigManager;

ConfigManager* config_manager_create(const char* config_path);
void config_manager_destroy(ConfigManager* cm);

int   config_get_int(const ConfigManager* cm, const char* key_path, int default_val);
float config_get_float(const ConfigManager* cm, const char* key_path, float default_val);
bool  config_get_bool(const ConfigManager* cm, const char* key_path, bool default_val);
const char* config_get_string(const ConfigManager* cm, const char* key_path, const char* default_val);

void config_set_int(ConfigManager* cm, const char* key_path, int val);
void config_set_float(ConfigManager* cm, const char* key_path, float val);
void config_set_bool(ConfigManager* cm, const char* key_path, bool val);
void config_set_string(ConfigManager* cm, const char* key_path, const char* val);

void config_load_defaults(ConfigManager* cm);
int  config_load_from_file(ConfigManager* cm, const char* path);
int  config_save_to_file(const ConfigManager* cm, const char* path);

#ifdef __cplusplus
}
#endif

#endif
