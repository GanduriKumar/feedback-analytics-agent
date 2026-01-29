# Backend Cleanup Guide

## Summary of Changes

The backend has been restructured, and several files are now redundant.

## Files Status

### ✅ KEEP (Still Needed)

1. **`backend/` directory** - New unified backend structure
2. **`chroma_db/` directory** - Vector database storage
3. **`docs/` directory** - Documentation
4. **`config/` directory** - Still used by root-level scripts (temporary)
5. **`.env`** - Environment configuration
6. **`requirements.txt`** - Root level dependencies (for backward compatibility)
7. **`README.md`** - Project overview
8. **`LICENSE`**
9. **`.gitignore`**

### ⚠️ OPTIONAL KEEP (CLI Tools - Your Choice)

These standalone CLI scripts might still be useful for command-line users:

1. **`review_analyzer_agent.py`** - Standalone CLI for running analysis
   - Useful for: Quick terminal-based analysis without API
   - Can be moved to `scripts/` directory if you want to keep it

### ❌ SAFE TO REMOVE (Superseded by Backend)

These files have been integrated into the unified backend:

1. **`a2acompatible_analyzer_agent.py`** → Merged into `backend/app/main.py`
2. **`custom_apis.py`** → Merged into `backend/app/main.py`
3. **`custom_pipeline.py`** → Copied to `backend/app/core/pipeline.py`
4. **`feedback_analyzer.py`** → Copied to `backend/app/core/analyzer.py`
5. **`query_vectorDB.py`** → Copied to `backend/app/core/vector_db.py`
6. **`src/` directory** → Copied to `backend/app/` (tools & utilities)

## Recommended Action Plan

### Option 1: Clean Removal (Recommended)

Remove all redundant files and commit to new structure:

```powershell
# Backup first (optional)
mkdir old_files
Move-Item a2acompatible_analyzer_agent.py, custom_apis.py, custom_pipeline.py, feedback_analyzer.py, query_vectorDB.py old_files/
Move-Item src old_files/

# Or delete directly
Remove-Item a2acompatible_analyzer_agent.py
Remove-Item custom_apis.py
Remove-Item custom_pipeline.py
Remove-Item feedback_analyzer.py
Remove-Item query_vectorDB.py
Remove-Item src -Recurse

# Keep review_analyzer_agent.py if you want CLI tool
# Or move it to scripts directory
mkdir scripts
Move-Item review_analyzer_agent.py scripts/
```

### Option 2: Create Archive (Conservative)

Keep old files in an archive directory for reference:

```powershell
mkdir legacy
Move-Item a2acompatible_analyzer_agent.py, custom_apis.py, custom_pipeline.py legacy/
Move-Item feedback_analyzer.py, query_vectorDB.py, review_analyzer_agent.py legacy/
Move-Item src legacy/
```

### Option 3: Git Branch (Most Conservative)

Create a branch with old structure before cleanup:

```powershell
git checkout -b pre-restructure-backup
git add .
git commit -m "Backup before cleanup"
git checkout main

# Then proceed with cleanup
```

## What About `config/` and `requirements.txt`?

### `config/` Directory
**Current Status**: In both root and `backend/config/`

**Recommendation**: 
- The `backend/config/` is the authoritative version
- Root `config/` can be removed OR made a symlink
- Update `.env` to point backend to use `backend/config/`

```powershell
# Remove root config (backend has its own copy)
Remove-Item config -Recurse

# OR create symlink (advanced)
# mklink /D config backend\config
```

### `requirements.txt`
**Current Status**: In both root and `backend/`

**Recommendation**: 
- Keep root `requirements.txt` for backward compatibility
- Point users to `backend/requirements.txt` in documentation
- Eventually remove root version

## After Cleanup - Updated Structure

```
feedback-analytics-agent/
├── backend/                 # ✅ All backend code
│   ├── app/
│   │   ├── main.py         # ✅ Unified API
│   │   ├── core/           # ✅ Business logic
│   │   ├── models/         # ✅ Schemas
│   │   ├── tools/          # ✅ Analysis tools
│   │   └── utilities/      # ✅ Helpers
│   ├── config/             # ✅ Configuration
│   ├── requirements.txt    # ✅ Dependencies
│   └── README.md
├── frontend/               # 🚧 To be created
├── docs/                   # ✅ Documentation
├── scripts/                # ⚠️ Optional: CLI tools
│   └── review_analyzer_agent.py
├── chroma_db/             # ✅ Database
├── .env                   # ✅ Environment
├── .gitignore
├── LICENSE
├── README.md
└── RESTRUCTURING.md       # 📄 This guide
```

## Verification Steps

After cleanup, verify everything works:

1. **Test Backend Startup:**
   ```powershell
   cd backend/app
   python main.py
   ```

2. **Test API Endpoints:**
   ```powershell
   # Health check
   curl http://localhost:8000/api/health
   
   # Capabilities
   curl http://localhost:8000/api/capabilities
   ```

3. **Check Imports:**
   ```powershell
   cd backend
   python -c "from app.core.analyzer import execute_graph_pipeline; print('OK')"
   python -c "from app.core.vector_db import query_vector_db; print('OK')"
   python -c "from app.core.pipeline import *; print('OK')"
   ```

4. **Test Full Pipeline:**
   ```powershell
   # Run the pipeline script from backend
   cd backend/app
   python core/pipeline.py
   ```

## Update .gitignore

Add to `.gitignore` if you create legacy/old_files directories:

```
# Legacy files (temporary)
legacy/
old_files/

# Keep ignoring these
chroma_db/
*.log
*.csv
*.json
!backend/config/*.csv
```

## Decision Matrix

| File | Keep? | Why |
|------|-------|-----|
| `a2acompatible_analyzer_agent.py` | ❌ No | Merged into `backend/app/main.py` |
| `custom_apis.py` | ❌ No | Merged into `backend/app/main.py` |
| `custom_pipeline.py` | ❌ No | Copied to `backend/app/core/pipeline.py` |
| `feedback_analyzer.py` | ❌ No | Copied to `backend/app/core/analyzer.py` |
| `query_vectorDB.py` | ❌ No | Copied to `backend/app/core/vector_db.py` |
| `review_analyzer_agent.py` | ⚠️ Maybe | Standalone CLI tool - move to `scripts/` |
| `src/` | ❌ No | Copied to `backend/app/` |
| `config/` (root) | ⚠️ Maybe | Backend has its own copy |
| `requirements.txt` (root) | ⚠️ Maybe | Keep for backward compatibility |

## My Recommendation

**Go with Option 1 (Clean Removal)** for these reasons:

1. ✅ Clearer project structure
2. ✅ No confusion about which files to use
3. ✅ Easier for new developers
4. ✅ Backend is self-contained
5. ✅ Frontend will have clean slate

**Execute this:**

```powershell
# Create scripts directory for CLI tools (optional)
New-Item -ItemType Directory -Path scripts -Force
Move-Item review_analyzer_agent.py scripts/

# Remove superseded files
Remove-Item a2acompatible_analyzer_agent.py, custom_apis.py, custom_pipeline.py
Remove-Item feedback_analyzer.py, query_vectorDB.py
Remove-Item src -Recurse -Force

# Clean up root config (backend has its own)
Remove-Item config -Recurse -Force

# Update README to point to new structure
Write-Host "✅ Cleanup complete! Test the backend:"
Write-Host "cd backend/app"
Write-Host "python main.py"
```

---

**Next Steps:**
1. Run the cleanup commands above
2. Test backend: `cd backend/app && python main.py`
3. Verify API docs: http://localhost:8000/api/docs
4. Proceed with React frontend development 🚀
