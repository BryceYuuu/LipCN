/* Relocatable in-process launcher: app permissions belong to LipCN, not Python. */
#include <Python.h>
#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void join(char *out, const char *base, const char *tail) {
    if (snprintf(out, PATH_MAX, "%s/%s", base, tail) >= PATH_MAX) {
        fputs("LipCN bundle path is too long\n", stderr);
        exit(2);
    }
}

int main(int argc, char **argv) {
    char executable[PATH_MAX], resolved[PATH_MAX], resources[PATH_MAX];
    uint32_t length = sizeof executable;
    if (_NSGetExecutablePath(executable, &length) != 0 || !realpath(executable, resolved)) return 2;
    char *slash = strrchr(resolved, '/');
    if (!slash) return 2;
    *slash = '\0';
    join(resources, resolved, "../Resources");
    char runtime[PATH_MAX], source[PATH_MAX], library[PATH_MAX], dynload[PATH_MAX], site[PATH_MAX];
    join(runtime, resources, "runtime");
    join(source, resources, "source");
    join(library, runtime, "lib/python3.12");
    join(dynload, library, "lib-dynload");
    join(site, library, "site-packages");
    setenv("LIPFLOW_APP", "1", 1);
    setenv("PYTHONUNBUFFERED", "1", 1);

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.parse_argv = 0;
    config.write_bytecode = 0;
    PyStatus status = PyConfig_SetBytesString(&config, &config.home, runtime);
    if (PyStatus_Exception(status)) goto error;
    status = PyConfig_SetBytesArgv(&config, argc, argv);
    if (PyStatus_Exception(status)) goto error;
    status = PyConfig_SetString(&config, &config.run_module, L"lipcn");
    if (PyStatus_Exception(status)) goto error;
    config.module_search_paths_set = 1;
    const char *paths[] = {source, library, dynload, site};
    for (int i = 0; i < 4; ++i) {
        wchar_t *path = Py_DecodeLocale(paths[i], NULL);
        if (!path) { PyConfig_Clear(&config); return 2; }
        status = PyWideStringList_Append(&config.module_search_paths, path);
        PyMem_RawFree(path);
        if (PyStatus_Exception(status)) goto error;
    }
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) goto error;
    PyConfig_Clear(&config);
    return Py_RunMain();
error:
    PyConfig_Clear(&config);
    Py_ExitStatusException(status);
}
