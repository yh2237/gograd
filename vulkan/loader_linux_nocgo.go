//go:build linux && !android && amd64 && !cgo

package vulkan

import "unsafe"

//go:cgo_import_dynamic x_pthread_create pthread_create "libpthread.so.0"
//go:cgo_import_dynamic x_pthread_self pthread_self "libpthread.so.0"
//go:cgo_import_dynamic x_pthread_getattr_np pthread_getattr_np "libpthread.so.0"
//go:cgo_import_dynamic x_pthread_attr_init pthread_attr_init "libpthread.so.0"
//go:cgo_import_dynamic x_pthread_attr_destroy pthread_attr_destroy "libpthread.so.0"
//go:cgo_import_dynamic x_pthread_attr_getstack pthread_attr_getstack "libpthread.so.0"
//go:cgo_import_dynamic x_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.so.0"
//go:cgo_import_dynamic x_pthread_sigmask pthread_sigmask "libpthread.so.0"
//go:cgo_import_dynamic x_malloc malloc "libc.so.6"
//go:cgo_import_dynamic x_free free "libc.so.6"
//go:cgo_import_dynamic x_sigfillset sigfillset "libc.so.6"
//go:cgo_import_dynamic x_abort abort "libc.so.6"
//go:cgo_import_dynamic x_setenv setenv "libc.so.6"
//go:cgo_import_dynamic x_unsetenv unsetenv "libc.so.6"

//go:linkname runtimeIsCgo runtime.iscgo
var runtimeIsCgo = true

//go:linkname cgoInitHook _cgo_init
var cgoInitHook = unsafe.Pointer(&cgoInitSym)

//go:linkname cgoThreadStartHook _cgo_thread_start
var cgoThreadStartHook = unsafe.Pointer(&cgoThreadStartSym)

//go:linkname cgoNotifyInitDoneHook _cgo_notify_runtime_init_done
var cgoNotifyInitDoneHook = unsafe.Pointer(&cgoNotifyInitDoneSym)

//go:linkname cgoPthreadKeyCreatedHook _cgo_pthread_key_created
var cgoPthreadKeyCreatedHook = unsafe.Pointer(&cgoPthreadKeyCreated)

//go:linkname cgoSetenvHook runtime._cgo_setenv
var cgoSetenvHook = unsafe.Pointer(&cgoSetenvSym)

//go:linkname cgoUnsetenvHook runtime._cgo_unsetenv
var cgoUnsetenvHook = unsafe.Pointer(&cgoUnsetenvSym)

//go:linkname setCrosscall2Hook runtime.set_crosscall2
var setCrosscall2Hook = func() {}

var (
	cgoInitSym           byte
	cgoThreadStartSym    byte
	cgoNotifyInitDoneSym byte
	cgoSetenvSym         byte
	cgoUnsetenvSym       byte
	cgoPthreadKeyCreated uintptr
)
