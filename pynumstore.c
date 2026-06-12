/// Copyright 2026 Theo Lincke
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
///     http://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.

#pragma once

// Python
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Numpy
#define PY_ARRAY_UNIQUE_SYMBOL _NUMSTORE_ARRAY_API
#define NPY_NO_DEPRECATED_API  NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

// System
#include <string.h>

// Numstore
#include "compiler.h"
#include "numstore.h"
#include "testing/testing.h"
#include "types.h"

/******************************************************************************
 * SECTION: Forward Declaration
 ******************************************************************************/

PyObject      *pyns_compile_type (PyObject *Py_UNUSED (m), PyObject *arg);
int            pyns_verify_types (PyArray_Descr *dtype, struct type *type);
PyArray_Descr *pyns_type_to_dtype (const struct type *t);

PyObject *pyns_db_open (PyObject *Py_UNUSED (m), PyObject *arg);
PyObject *pyns_db_close (PyObject *Py_UNUSED (m), PyObject *arg);

PyObject *pyns_db_begin (PyObject *Py_UNUSED (m), PyObject *arg);
PyObject *pyns_txn_commit (PyObject *Py_UNUSED (m), PyObject *arg);
PyObject *pyns_txn_rollback (PyObject *Py_UNUSED (m), PyObject *arg);

PyObject *pyns_var_create (PyObject *Py_UNUSED (m), PyObject *args);
PyObject *pyns_var_delete (PyObject *Py_UNUSED (m), PyObject *args);
PyObject *pyns_var_len (PyObject *Py_UNUSED (m), PyObject *args);
PyObject *pyns_var_exists (PyObject *Py_UNUSED (m), PyObject *args);

PyObject *pyns_var_read (PyObject *Py_UNUSED (m), PyObject *args);
PyObject *pyns_var_insert (PyObject *Py_UNUSED (m), PyObject *args);
PyObject *pyns_var_write (PyObject *Py_UNUSED (m), PyObject *args);
PyObject *pyns_var_remove (PyObject *Py_UNUSED (m), PyObject *args);

static const char DB_CAPSULE[]  = "numstore.db";
static const char TXN_CAPSULE[] = "numstore.txn";

static inline void
_nspy_release_db (PyObject *obj)
{
  nsdb_t *ns = (nsdb_t *)PyCapsule_GetPointer (obj, DB_CAPSULE);
  ASSERT (ns);
  nsdb_close (ns);
}

static inline nsdb_t *
_unwrap_db (PyObject *db)
{
  return (nsdb_t *)PyCapsule_GetPointer (db, DB_CAPSULE);
}

// Returns nsdb_t * from txn capsule, or NULL (without setting error) if None.
static inline nsdb_t *
_unwrap_txn (PyObject *txn_or_none)
{
  if (txn_or_none == Py_None)
  {
    return NULL;
  }
  return (nsdb_t *)PyCapsule_GetPointer (txn_or_none, TXN_CAPSULE);
}

// Returns the active nsdb_t *: from txn if present, otherwise from db.
static inline nsdb_t *
_active_ns (PyObject *db, PyObject *txn_or_none)
{
  if (txn_or_none != Py_None)
  {
    return (nsdb_t *)PyCapsule_GetPointer (txn_or_none, TXN_CAPSULE);
  }
  return (nsdb_t *)PyCapsule_GetPointer (db, DB_CAPSULE);
}

// Sets a Python RuntimeError from the nsdb error string.
static inline void
_pyns_set_error (nsdb_t *ns)
{
  const char *err = nsdb_strerror (ns);
  if (err)
  {
    PyErr_SetString (PyExc_RuntimeError, err);
  }
  else
  {
    PyErr_SetString (PyExc_RuntimeError, "numstore operation failed");
  }
}

static inline Py_ssize_t
elsize (PyArray_Descr *type)
{
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
  return (type)->elsize;
#else
  return PyDataType_ELSIZE (type);
#endif
}

// Build a complex valued struct
static PyArray_Descr *
build_complex_struct (int component_typenum)
{
  PyArray_Descr *comp   = NULL; // dtype for the single component (e.g. f32)
  PyObject      *fields = NULL; // the list of fields
  PyArray_Descr *out =
      NULL; // dtype of the output complex float (a struct of re im)
  PyObject *name = NULL;
  PyObject *tup  = NULL;

  // Internal type
  comp   = PyArray_DescrFromType (component_typenum);
  fields = PyList_New (2);

  if (comp == NULL || fields == NULL)
  {
    goto fail;
  }

  // Names
  static const char *names[2] = {"re", "im"};
  for (int i = 0; i < 2; i++)
  {
    // Generate python string
    name = PyUnicode_FromString (names[i]);
    tup  = PyTuple_New (2);

    if (name == NULL || tup == NULL)
    {
      goto fail;
    }

    // increase ref so that SET_ITEM doesn't set count to 0
    Py_INCREF (comp);

    // All these Steal
    PyTuple_SET_ITEM (tup, 0, name);
    PyTuple_SET_ITEM (tup, 1, (PyObject *)comp);
    PyList_SET_ITEM (fields, i, tup);

    name = NULL;
    tup  = NULL;
  }

  if (PyArray_DescrConverter (fields, &out) != NPY_SUCCEED)
  {
    goto fail;
  }

  Py_DECREF (comp);
  Py_DECREF (fields);

  return out;

fail:
  Py_XDECREF (name);
  Py_XDECREF (tup);
  Py_XDECREF (comp);
  Py_XDECREF (fields);
  return NULL;
}

static PyArray_Descr *
primitive_to_dtype (enum prim_t p)
{
  int typenum;
  switch (p)
  {
    case U8: typenum = NPY_UINT8; break;
    case U16: typenum = NPY_UINT16; break;
    case U32: typenum = NPY_UINT32; break;
    case U64: typenum = NPY_UINT64; break;
    case I8: typenum = NPY_INT8; break;
    case I16: typenum = NPY_INT16; break;
    case I32: typenum = NPY_INT32; break;
    case I64: typenum = NPY_INT64; break;
    case F16: typenum = NPY_FLOAT16; break;
    case F32: typenum = NPY_FLOAT32; break;
    case F64: typenum = NPY_FLOAT64; break;
    case F128: typenum = NPY_LONGDOUBLE; break;
    case CF64: typenum = NPY_COMPLEX64; break;
    case CF128: typenum = NPY_COMPLEX128; break;
    case CF256: typenum = NPY_CLONGDOUBLE; break;
    case CF32: return build_complex_struct (NPY_FLOAT16);
    case CI16: return build_complex_struct (NPY_INT8);
    case CI32: return build_complex_struct (NPY_INT16);
    case CI64: return build_complex_struct (NPY_INT32);
    case CI128: return build_complex_struct (NPY_INT64);
    case CU16: return build_complex_struct (NPY_UINT8);
    case CU32: return build_complex_struct (NPY_UINT16);
    case CU64: return build_complex_struct (NPY_UINT32);
    case CU128: return build_complex_struct (NPY_UINT64);
    default:
      PyErr_Format (PyExc_ValueError, "unknown numstore primitive: %d", (int)p);
      return NULL;
  }

  PyArray_Descr *d = PyArray_DescrFromType (typenum);
  if (d == NULL)
  {
    return NULL;
  }

  return d;
}

static PyArray_Descr *
struct_to_dtype (const struct struct_t *st)
{
  ASSERT (st->len > 0);
  PyObject      *fields = NULL; // List of fields
  PyObject      *name   = NULL; // name of each field
  PyArray_Descr *sub    = NULL; // sub type of each field
  PyObject      *tup    = NULL; // the wrapper of (name, sub)
  PyArray_Descr *out    = NULL; // The result

  fields = PyList_New (st->len);
  if (fields == NULL)
  {
    goto fail;
  }

  for (u16 i = 0; i < st->len; i++)
  {
    name = PyUnicode_FromStringAndSize (
        st->keys[i].data,
        (Py_ssize_t)st->keys[i].len
    );
    sub = pyns_type_to_dtype (st->types[i]);
    tup = PyTuple_New (2);

    if (name == NULL || sub == NULL || tup == NULL)
    {
      goto fail;
    }

    PyTuple_SET_ITEM (tup, 0, name);
    name = NULL;
    PyTuple_SET_ITEM (tup, 1, (PyObject *)sub);
    sub = NULL;
    PyList_SET_ITEM (fields, i, tup);
    tup = NULL;
  }

  if (PyArray_DescrConverter (fields, &out) != NPY_SUCCEED)
  {
    goto fail;
  }

  Py_DECREF (fields);

  return out;

fail:
  Py_XDECREF (name);
  Py_XDECREF (sub);
  Py_XDECREF (tup);
  Py_XDECREF (fields);
  return NULL;
}

static PyArray_Descr *
union_to_dtype (const struct union_t *un)
{
  ASSERT (un->len > 0);
  PyObject      *names    = NULL; // names
  PyObject      *formats  = NULL;
  PyObject      *offsets  = NULL;
  PyObject      *name     = NULL;
  PyArray_Descr *sub      = NULL;
  PyObject      *off      = NULL;
  PyObject      *itemsize = NULL;
  PyObject      *spec     = NULL;
  PyArray_Descr *out      = NULL;

  names   = PyList_New (un->len);
  formats = PyList_New (un->len);
  offsets = PyList_New (un->len);
  if (names == NULL || formats == NULL || offsets == NULL)
  {
    goto fail;
  }

  Py_ssize_t max_size = 0;
  for (u16 i = 0; i < un->len; i++)
  {
    name = PyUnicode_FromStringAndSize (
        un->keys[i].data,
        (Py_ssize_t)un->keys[i].len
    );
    sub = pyns_type_to_dtype (un->types[i]);
    off = PyLong_FromLong (0);

    if (name == NULL || sub == NULL || off == NULL)
    {
      goto fail;
    }

    Py_ssize_t isize = elsize (sub);

    if (isize > max_size)
    {
      max_size = isize;
    }

    PyList_SET_ITEM (names, i, name);
    name = NULL;
    PyList_SET_ITEM (formats, i, (PyObject *)sub);
    sub = NULL;
    PyList_SET_ITEM (offsets, i, off);
    off = NULL;
  }

  itemsize = PyLong_FromSsize_t (max_size);
  spec     = PyDict_New ();

  if (itemsize == NULL || spec == NULL)
  {
    goto fail;
  }

  if (PyDict_SetItemString (spec, "names", names) != 0)
  {
    goto fail;
  }
  if (PyDict_SetItemString (spec, "formats", formats) != 0)
  {
    goto fail;
  }
  if (PyDict_SetItemString (spec, "offsets", offsets) != 0)
  {
    goto fail;
  }
  if (PyDict_SetItemString (spec, "itemsize", itemsize) != 0)
  {
    goto fail;
  }

  Py_DECREF (names);
  names = NULL;
  Py_DECREF (formats);
  formats = NULL;
  Py_DECREF (offsets);
  offsets = NULL;
  Py_DECREF (itemsize);
  itemsize = NULL;

  if (PyArray_DescrConverter (spec, &out) != NPY_SUCCEED)
  {
    goto fail;
  }

  Py_DECREF (spec);
  return out;

fail:
  Py_XDECREF (name);
  Py_XDECREF (sub);
  Py_XDECREF (off);
  Py_XDECREF (names);
  Py_XDECREF (formats);
  Py_XDECREF (offsets);
  Py_XDECREF (itemsize);
  Py_XDECREF (spec);
  return NULL;
}

static PyArray_Descr *
sarray_to_dtype (const struct sarray_t *sa)
{
  ASSERT (sa->rank > 0);
  PyArray_Descr *sub   = NULL;
  PyObject      *shape = NULL;
  PyObject      *d     = NULL;
  PyObject      *spec  = NULL;
  PyArray_Descr *out   = NULL;

  sub   = pyns_type_to_dtype (sa->t);
  shape = PyTuple_New (sa->rank);

  if (sub == NULL || shape == NULL)
  {
    goto fail;
  }

  for (u16 i = 0; i < sa->rank; i++)
  {
    d = PyLong_FromUnsignedLong ((unsigned long)sa->dims[i]);

    if (d == NULL)
    {
      goto fail;
    }

    PyTuple_SET_ITEM (shape, i, d);
    d = NULL;
  }

  spec = PyTuple_New (2);
  if (spec == NULL)
  {
    goto fail;
  }

  PyTuple_SET_ITEM (spec, 0, (PyObject *)sub);
  sub = NULL;
  PyTuple_SET_ITEM (spec, 1, shape);
  shape = NULL;

  if (PyArray_DescrConverter (spec, &out) != NPY_SUCCEED)
  {
    goto fail;
  }

  Py_DECREF (spec);
  return out;

fail:
  Py_XDECREF (d);
  Py_XDECREF (sub);
  Py_XDECREF (shape);
  Py_XDECREF (spec);
  return NULL;
}

PyArray_Descr *
pyns_type_to_dtype (const struct type *t)
{
  ASSERT (t);

  switch (t->type)
  {
    case T_PRIM: return primitive_to_dtype (t->p);
    case T_STRUCT: return struct_to_dtype (&t->st);
    case T_UNION: return union_to_dtype (&t->un);
    case T_SARRAY: return sarray_to_dtype (&t->sa);
    default:
    {
      UNREACHABLE ();
    }
  }
}

PyObject *
pyns_compile_type (PyObject *Py_UNUSED (m), PyObject *arg)
{
  const char *src = PyUnicode_AsUTF8 (arg);
  if (!src)
  {
    return NULL;
  }

  struct type        t;
  struct chunk_alloc temp;
  chunk_alloc_create_default (&temp);
  error e = error_create ();

  if (compile_type (&t, src, &temp, &e))
  {
    PyErr_Format (PyExc_ValueError, "Error: %.*s", e.cmlen, e.cause_msg);
    chunk_alloc_free_all (&temp);
    return NULL;
  }

  PyArray_Descr *ret = pyns_type_to_dtype (&t);
  chunk_alloc_free_all (&temp);

  return (PyObject *)ret;
}

PyObject *
pyns_db_begin (PyObject *Py_UNUSED (m), PyObject *arg)
{
  // Get the wrapped database
  nsdb_t *ns = _unwrap_db (arg);
  if (!ns)
  {
    return NULL;
  }

  // BEGIN TXN
  if (nsdb_begin (ns) < 0)
  {
    _pyns_set_error (ns);
    return NULL;
  }

  return PyCapsule_New ((void *)ns, TXN_CAPSULE, NULL);
}

PyObject *
pyns_db_close (PyObject *Py_UNUSED (m), PyObject *arg)
{
  nsdb_t *ns = _unwrap_db (arg);
  if (!ns)
  {
    return NULL;
  }

  PyCapsule_SetDestructor (arg, NULL);

  if (nsdb_close (ns) < 0)
  {
    PyErr_SetString (PyExc_RuntimeError, "Failed to close numstore database");
    return NULL;
  }

  Py_RETURN_NONE;
}

PyObject *
pyns_db_open (PyObject *Py_UNUSED (m), PyObject *arg)
{
  if (!PyUnicode_Check (arg))
  {
    PyErr_SetString (PyExc_TypeError, "path must be str");
    return NULL;
  }

  const char *path = PyUnicode_AsUTF8 (arg);
  if (!path)
  {
    return NULL;
  }

  nsdb_t *ns = nsdb_open (path);
  if (!ns)
  {
    PyErr_SetString (PyExc_RuntimeError, "Failed to open numstore database");
    return NULL;
  }

  return PyCapsule_New ((void *)ns, DB_CAPSULE, _nspy_release_db);
}

PyObject *
pyns_txn_commit (PyObject *Py_UNUSED (m), PyObject *arg)
{
  nsdb_t *ns = (nsdb_t *)PyCapsule_GetPointer (arg, TXN_CAPSULE);
  if (!ns)
  {
    return NULL;
  }

  // COMMIT
  if (nsdb_commit (ns) < 0)
  {
    _pyns_set_error (ns);
    return NULL;
  }

  Py_RETURN_NONE;
}

PyObject *
pyns_txn_rollback (PyObject *Py_UNUSED (m), PyObject *arg)
{
  nsdb_t *ns = (nsdb_t *)PyCapsule_GetPointer (arg, TXN_CAPSULE);
  if (!ns)
  {
    return NULL;
  }

  // ROLLBACK
  if (nsdb_rollback (ns) < 0)
  {
    _pyns_set_error (ns);
    return NULL;
  }

  Py_RETURN_NONE;
}

// var_create(db, txn_or_none, name: str, type_str: str) -> None
PyObject *
pyns_var_create (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject   *db;
  PyObject   *txn_or_none;
  const char *name;
  const char *type_str;

  if (!PyArg_ParseTuple (args, "OOss", &db, &txn_or_none, &name, &type_str))
  {
    return NULL;
  }

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    return NULL;
  }

  if (nsdb_create (ns, name, type_str) < 0)
  {
    _pyns_set_error (ns);
    return NULL;
  }

  Py_RETURN_NONE;
}

PyObject *
pyns_var_delete (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject   *db;
  PyObject   *txn_or_none;
  const char *name;

  if (!PyArg_ParseTuple (args, "OOs", &db, &txn_or_none, &name))
  {
    return NULL;
  }

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    return NULL;
  }

  if (nsdb_delete (ns, name) < 0)
  {
    _pyns_set_error (ns);
    return NULL;
  }

  Py_RETURN_NONE;
}

PyObject *
pyns_var_exists (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject   *db          = NULL;
  PyObject   *txn_or_none = NULL;
  const char *name        = NULL;
  nsdb_var_t *var         = NULL;

  if (!PyArg_ParseTuple (args, "OOs", &db, &txn_or_none, &name))
  {
    goto fail;
  }

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    goto fail;
  }

  if (nsdb_get_if_exists (ns, &var, name))
  {
    goto fail;
  }

  bool exists = var != NULL;
  if (var)
  {
    nsdb_free (var);
  }

  return PyBool_FromLong (exists);

fail:
  nsdb_free (var);
  return NULL;
}

PyObject *
pyns_var_insert (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject   *db          = NULL;
  PyObject   *txn_or_none = NULL;
  PyObject   *ofst_obj    = NULL;
  PyObject   *data_obj    = NULL;
  const char *name        = NULL;
  nsdb_var_t *var         = NULL;

  if (!PyArg_ParseTuple (
          args,
          "OOsOO",
          &db,
          &txn_or_none,
          &name,
          &ofst_obj,
          &data_obj
      ))
  {
    goto fail;
  }

  if (!PyLong_Check (ofst_obj))
  {
    PyErr_SetString (PyExc_TypeError, "offset must be int");
    goto fail;
  }

  long long ofst = PyLong_AsLongLong (ofst_obj);
  if (ofst == -1 && PyErr_Occurred ())
  {
    goto fail;
  }

  if (!PyArray_Check (data_obj))
  {
    PyErr_SetString (PyExc_TypeError, "data must be a numpy array");
    goto fail;
  }

  PyArrayObject *arr    = (PyArrayObject *)data_obj;
  void          *buf    = PyArray_DATA (arr);
  npy_intp       nelems = PyArray_SIZE (arr);

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    goto fail;
  }

  var = nsdb_get (ns, name);
  if (var == NULL)
  {
    goto fail;
  }

  if (pyns_verify_types (PyArray_DESCR (arr), var->var.dtype) != 0)
  {
    goto fail;
  }

  sb_size inserted = nsdb_insert (ns, var, buf, (sb_size)ofst, (b_size)nelems);
  if (inserted < 0)
  {
    _pyns_set_error (ns);
    goto fail;
  }

  nsdb_free (var);

  Py_RETURN_NONE;

fail:
  nsdb_free (var);
  return NULL;
}

PyObject *
pyns_var_len (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject   *db;
  PyObject   *txn_or_none;
  const char *name;

  if (!PyArg_ParseTuple (args, "OOs", &db, &txn_or_none, &name))
  {
    return NULL;
  }

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    return NULL;
  }

  sb_size len = nsdb_len (ns, name);
  if (len < 0)
  {
    _pyns_set_error (ns);
    return NULL;
  }

  return PyLong_FromSsize_t ((Py_ssize_t)len);
}

PyObject *
pyns_var_read (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject      *db          = NULL;
  PyObject      *txn_or_none = NULL;
  PyObject      *key_obj     = NULL;
  PyObject      *r_start     = NULL;
  PyObject      *r_stop      = NULL;
  PyObject      *r_step      = NULL;
  PyObject      *arr         = NULL;
  PyArray_Descr *dtype       = NULL;
  nsdb_var_t    *var         = NULL;
  void          *buf         = NULL;
  const char    *name        = NULL;

  if (!PyArg_ParseTuple (args, "OOsO", &db, &txn_or_none, &name, &key_obj))
  {
    goto fail;
  }

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    goto fail;
  }

  long long start = 0;
  long long step  = 1;
  long long stop  = 0;
  int       flags = 0;

  if (PyLong_Check (key_obj))
  {
    start = PyLong_AsLongLong (key_obj);
    if (start == -1 && PyErr_Occurred ())
    {
      goto fail;
    }
    step  = 1;
    stop  = start + 1;
    flags = START_PRESENT | STOP_PRESENT | STEP_PRESENT;
  }
  else
  {
    r_start = PyObject_GetAttrString (key_obj, "start");
    r_stop  = PyObject_GetAttrString (key_obj, "stop");
    r_step  = PyObject_GetAttrString (key_obj, "step");
    if (!r_start || !r_stop || !r_step)
    {
      goto fail;
    }

    flags = COLON_PRESENT;

    if (r_start != Py_None)
    {
      start = PyLong_AsLongLong (r_start);
      if (start == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= START_PRESENT;
    }
    if (r_stop != Py_None)
    {
      stop = PyLong_AsLongLong (r_stop);
      if (stop == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= STOP_PRESENT;
    }
    if (r_step != Py_None)
    {
      step = PyLong_AsLongLong (r_step);
      if (step == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= STEP_PRESENT;
    }

    Py_DECREF (r_start);
    r_start = NULL;
    Py_DECREF (r_stop);
    r_stop = NULL;
    Py_DECREF (r_step);
    r_step = NULL;
  }

  if (!(flags & START_PRESENT))
  {
    start = 0;
    flags |= START_PRESENT;
  }
  if (!(flags & STEP_PRESENT))
  {
    step = 1;
    flags |= STEP_PRESENT;
  }
  if (!(flags & STOP_PRESENT))
  {
    sb_size len = nsdb_len (ns, name);
    if (len < 0)
    {
      _pyns_set_error (ns);
      goto fail;
    }
    stop = (long long)len;
    flags |= STOP_PRESENT;
  }

  if (step <= 0)
  {
    PyErr_SetString (PyExc_ValueError, "key step must be positive");
    goto fail;
  }

  var = nsdb_get (ns, name);
  if (var == NULL)
  {
    goto fail;
  }

  dtype = (PyArray_Descr *)pyns_type_to_dtype (var->var.dtype);
  if (dtype == NULL)
  {
    goto fail;
  }

#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
  npy_intp tsize = dtype->elsize;
#else
  npy_intp tsize = (npy_intp)PyDataType_ELSIZE (dtype);
#endif

  npy_intp nelems_max;
  if (stop <= start)
  {
    nelems_max = 0;
  }
  else
  {
    nelems_max = (npy_intp)((stop - start + step - 1) / step);
  }

  if (nelems_max > 0)
  {
    buf = malloc ((size_t)(nelems_max * tsize));
    if (!buf)
    {
      PyErr_NoMemory ();
      goto fail;
    }
  }

  sb_size bytes_read = nsdb_read (
      ns,
      var,
      buf,
      (sb_size)start,
      (sb_size)step,
      (sb_size)stop,
      flags
  );
  if (bytes_read < 0)
  {
    _pyns_set_error (ns);
    goto fail;
  }

  npy_intp nelems_actual = (npy_intp)bytes_read;
  npy_intp dims[1]       = {nelems_actual};

  arr =
      PyArray_NewFromDescr (&PyArray_Type, dtype, 1, dims, NULL, NULL, 0, NULL);
  dtype = NULL; /* stolen by PyArray_NewFromDescr */
  if (!arr)
  {
    goto fail;
  }

  if (nelems_actual > 0)
  {
    memcpy (
        PyArray_DATA ((PyArrayObject *)arr),
        buf,
        (size_t)(nelems_actual * tsize)
    );
  }

  free (buf);
  nsdb_free (var);
  return arr;

fail:
  Py_XDECREF (r_start);
  Py_XDECREF (r_stop);
  Py_XDECREF (r_step);
  Py_XDECREF (arr);
  Py_XDECREF (dtype);
  free (buf);
  nsdb_free (var);
  return NULL;
}

PyObject *
pyns_var_remove (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject      *db          = NULL;
  PyObject      *txn_or_none = NULL;
  PyObject      *key_obj     = NULL;
  PyObject      *r_start     = NULL;
  PyObject      *r_stop      = NULL;
  PyObject      *r_step      = NULL;
  PyObject      *arr         = NULL;
  PyArray_Descr *dtype       = NULL;
  nsdb_var_t    *var         = NULL;
  void          *buf         = NULL;
  const char    *name        = NULL;

  if (!PyArg_ParseTuple (args, "OOsO", &db, &txn_or_none, &name, &key_obj))
  {
    goto fail;
  }

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    goto fail;
  }

  long long start = 0;
  long long step  = 1;
  long long stop  = 0;
  int       flags = 0;

  if (PyLong_Check (key_obj))
  {
    start = PyLong_AsLongLong (key_obj);
    if (start == -1 && PyErr_Occurred ())
    {
      goto fail;
    }
    step  = 1;
    stop  = start + 1;
    flags = START_PRESENT | STOP_PRESENT | STEP_PRESENT;
  }
  else
  {
    r_start = PyObject_GetAttrString (key_obj, "start");
    r_stop  = PyObject_GetAttrString (key_obj, "stop");
    r_step  = PyObject_GetAttrString (key_obj, "step");
    if (!r_start || !r_stop || !r_step)
    {
      goto fail;
    }

    flags = COLON_PRESENT;

    if (r_start != Py_None)
    {
      start = PyLong_AsLongLong (r_start);
      if (start == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= START_PRESENT;
    }
    if (r_stop != Py_None)
    {
      stop = PyLong_AsLongLong (r_stop);
      if (stop == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= STOP_PRESENT;
    }
    if (r_step != Py_None)
    {
      step = PyLong_AsLongLong (r_step);
      if (step == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= STEP_PRESENT;
    }

    Py_DECREF (r_start);
    r_start = NULL;
    Py_DECREF (r_stop);
    r_stop = NULL;
    Py_DECREF (r_step);
    r_step = NULL;
  }

  if (!(flags & START_PRESENT))
  {
    start = 0;
    flags |= START_PRESENT;
  }
  if (!(flags & STEP_PRESENT))
  {
    step = 1;
    flags |= STEP_PRESENT;
  }
  if (!(flags & STOP_PRESENT))
  {
    sb_size len = nsdb_len (ns, name);
    if (len < 0)
    {
      _pyns_set_error (ns);
      goto fail;
    }
    stop = (long long)len;
    flags |= STOP_PRESENT;
  }

  if (step <= 0)
  {
    PyErr_SetString (PyExc_ValueError, "key step must be positive");
    goto fail;
  }

  var = nsdb_get (ns, name);
  if (var == NULL)
  {
    goto fail;
  }

  dtype = (PyArray_Descr *)pyns_type_to_dtype (var->var.dtype);
  if (dtype == NULL)
  {
    goto fail;
  }

#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
  npy_intp tsize = dtype->elsize;
#else
  npy_intp tsize = (npy_intp)PyDataType_ELSIZE (dtype);
#endif

  npy_intp nelems_max;
  if (stop <= start)
  {
    nelems_max = 0;
  }
  else
  {
    nelems_max = (npy_intp)((stop - start + step - 1) / step);
  }

  if (nelems_max > 0)
  {
    buf = malloc ((size_t)(nelems_max * tsize));
    if (!buf)
    {
      PyErr_NoMemory ();
      goto fail;
    }
  }

  sb_size bytes_removed = nsdb_remove (
      ns,
      var,
      buf,
      (sb_size)start,
      (sb_size)step,
      (sb_size)stop,
      flags
  );
  if (bytes_removed < 0)
  {
    _pyns_set_error (ns);
    goto fail;
  }

  npy_intp nelems_actual = (npy_intp)bytes_removed;
  npy_intp dims[1]       = {nelems_actual};

  arr =
      PyArray_NewFromDescr (&PyArray_Type, dtype, 1, dims, NULL, NULL, 0, NULL);
  dtype = NULL; /* stolen by PyArray_NewFromDescr */
  if (!arr)
  {
    goto fail;
  }

  if (nelems_actual > 0)
  {
    memcpy (
        PyArray_DATA ((PyArrayObject *)arr),
        buf,
        (size_t)(nelems_actual * tsize)
    );
  }

  free (buf);
  nsdb_free (var);
  return arr;

fail:
  Py_XDECREF (r_start);
  Py_XDECREF (r_stop);
  Py_XDECREF (r_step);
  Py_XDECREF (arr);
  Py_XDECREF (dtype);
  free (buf);
  nsdb_free (var);
  return NULL;
}

PyObject *
pyns_var_write (PyObject *Py_UNUSED (m), PyObject *args)
{
  PyObject   *db          = NULL;
  PyObject   *txn_or_none = NULL;
  PyObject   *key_obj     = NULL;
  PyObject   *data_obj    = NULL;
  PyObject   *r_start     = NULL;
  PyObject   *r_stop      = NULL;
  PyObject   *r_step      = NULL;
  nsdb_var_t *var         = NULL;
  const char *name        = NULL;

  if (!PyArg_ParseTuple (
          args,
          "OOsOO",
          &db,
          &txn_or_none,
          &name,
          &key_obj,
          &data_obj
      ))
  {
    goto fail;
  }

  if (!PyArray_Check (data_obj))
  {
    PyErr_SetString (PyExc_TypeError, "data must be a numpy array");
    goto fail;
  }

  PyArrayObject *arr = (PyArrayObject *)data_obj;
  void          *buf = PyArray_DATA (arr);

  long long start = 0;
  long long step  = 1;
  long long stop  = 0;
  int       flags = 0;

  if (PyLong_Check (key_obj))
  {
    start = PyLong_AsLongLong (key_obj);
    if (start == -1 && PyErr_Occurred ())
    {
      goto fail;
    }
    step  = 1;
    stop  = start + 1;
    flags = START_PRESENT | STOP_PRESENT | STEP_PRESENT;
  }
  else
  {
    r_start = PyObject_GetAttrString (key_obj, "start");
    r_stop  = PyObject_GetAttrString (key_obj, "stop");
    r_step  = PyObject_GetAttrString (key_obj, "step");
    if (!r_start || !r_stop || !r_step)
    {
      goto fail;
    }

    flags = COLON_PRESENT;

    if (r_start != Py_None)
    {
      start = PyLong_AsLongLong (r_start);
      if (start == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= START_PRESENT;
    }
    if (r_stop != Py_None)
    {
      stop = PyLong_AsLongLong (r_stop);
      if (stop == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= STOP_PRESENT;
    }
    if (r_step != Py_None)
    {
      step = PyLong_AsLongLong (r_step);
      if (step == -1 && PyErr_Occurred ())
      {
        goto fail;
      }
      flags |= STEP_PRESENT;
    }

    Py_DECREF (r_start);
    r_start = NULL;
    Py_DECREF (r_stop);
    r_stop = NULL;
    Py_DECREF (r_step);
    r_step = NULL;
  }

  nsdb_t *ns = _active_ns (db, txn_or_none);
  if (!ns)
  {
    goto fail;
  }

  if (!(flags & START_PRESENT))
  {
    start = 0;
    flags |= START_PRESENT;
  }
  if (!(flags & STEP_PRESENT))
  {
    step = 1;
    flags |= STEP_PRESENT;
  }
  if (!(flags & STOP_PRESENT))
  {
    sb_size len = nsdb_len (ns, name);
    if (len < 0)
    {
      _pyns_set_error (ns);
      goto fail;
    }
    stop = (long long)len;
    flags |= STOP_PRESENT;
  }

  if (step <= 0)
  {
    PyErr_SetString (PyExc_ValueError, "key step must be positive");
    goto fail;
  }

  var = nsdb_get (ns, name);
  if (var == NULL)
  {
    goto fail;
  }

  if (pyns_verify_types (PyArray_DESCR (arr), var->var.dtype) != 0)
  {
    goto fail;
  }

  sb_size written = nsdb_write (
      ns,
      var,
      buf,
      (sb_size)start,
      (sb_size)step,
      (sb_size)stop,
      flags
  );
  if (written < 0)
  {
    _pyns_set_error (ns);
    goto fail;
  }

  nsdb_free (var);
  Py_RETURN_NONE;

fail:
  Py_XDECREF (r_start);
  Py_XDECREF (r_stop);
  Py_XDECREF (r_step);
  nsdb_free (var);
  return NULL;
}

int
pyns_verify_types (PyArray_Descr *dtype, struct type *type)
{
  PyArray_Descr *nsdtype = (PyArray_Descr *)pyns_type_to_dtype (type);
  if (nsdtype == NULL)
  {
    return -1;
  }

  int eq = PyArray_EquivTypes (dtype, nsdtype);
  Py_DECREF (nsdtype);

  if (!eq)
  {
    PyErr_SetString (
        PyExc_ValueError,
        "array dtype does not match variable type"
    );
    return -1;
  }

  return 0;
}

/******************************************************************************
 * SECTION: Module Code
 * ----------------------------------------------------------------------------
 * @brief
 *
 *
 ******************************************************************************/

static PyMethodDef pynumstore_methods[] = {
    // Utils
    {
        "ns_to_np",
        pyns_compile_type,
        METH_O,
        "ns_to_np(str) -> np.dtype",
    },

    // Lifecycle
    {
        "db_open",
        pyns_db_open,
        METH_O,
        "db_open(path) -> capsule",
    },
    {
        "db_close",
        pyns_db_close,
        METH_O,
        "db_close(db) -> None",
    },

    // Transactions
    {
        "db_begin",
        pyns_db_begin,
        METH_O,
        "db_begin(db) -> capsule",
    },
    {
        "txn_commit",
        pyns_txn_commit,
        METH_O,
        "txn_commit(txn) -> None",
    },
    {
        "txn_rollback",
        pyns_txn_rollback,
        METH_O,
        "txn_rollback(txn) -> None",
    },

    // Variable management
    {
        "var_create",
        pyns_var_create,
        METH_VARARGS,
        "var_create(db, txn_or_none, name, type_str) -> None",
    },
    {
        "var_delete",
        pyns_var_delete,
        METH_VARARGS,
        "var_delete(db, txn_or_none, name) -> None",
    },
    {
        "var_len",
        pyns_var_len,
        METH_VARARGS,
        "var_len(db, txn_or_none, var) -> int",
    },

    // Main Methods
    {
        "var_read",
        pyns_var_read,
        METH_VARARGS,
        "var_read(db, txn_or_none, var, key) -> NDArray",
    },
    {
        "var_insert",
        pyns_var_insert,
        METH_VARARGS,
        "var_insert(db, txn_or_none, var, ofst, data) -> None",
    },
    {
        "var_write",
        pyns_var_write,
        METH_VARARGS,
        "var_write(db, txn_or_none, var, key, data) -> None",
    },
    {
        "var_remove",
        pyns_var_remove,
        METH_VARARGS,
        "var_remove(db, txn_or_none, var, key) -> NDArray",
    },
    {
        "var_exists",
        pyns_var_exists,
        METH_VARARGS,
        "var_exists(db, txn_or_none, var) -> bool",
    },

    // End
    {NULL, NULL, 0, NULL},
};

static PyModuleDef pynumstore_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_pynumstore",
    .m_doc =
        "Thin C wrapper around smfile operations for the pynumstore package.",
    .m_size    = -1,
    .m_methods = pynumstore_methods,
};

PyMODINIT_FUNC PyInit__pynumstore (void);

PyMODINIT_FUNC
PyInit__pynumstore (void)
{
  import_array ();
  return PyModule_Create (&pynumstore_module);
}
