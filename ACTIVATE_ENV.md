# How to Activate Virtual Environment on Windows

## Quick Fix: Use Command Prompt Instead

**Easiest solution:** Open **Command Prompt** (cmd.exe) instead of PowerShell:

```cmd
cd C:\Users\dylan\Downloads\pineappleRL-Optimization\pineappleRL-Optimization
.venv\Scripts\activate.bat
```

This will work immediately without changing any security settings.

---

## If You Want to Use PowerShell

### Option 1: Temporary Fix (Current Session Only)
Run this command in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Then activate:
```powershell
.venv\Scripts\activate
```

**Note:** This only works for the current PowerShell session. You'll need to run it again each time you open a new PowerShell window.

### Option 2: Permanent Fix (Recommended)
1. Open PowerShell as **Administrator** (Right-click PowerShell → "Run as Administrator")
2. Run this command:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
3. Close and reopen PowerShell
4. Now you can activate normally:
```powershell
.venv\Scripts\activate
```

### Option 3: Bypass for Single Command
```powershell
powershell -ExecutionPolicy Bypass -File .venv\Scripts\Activate.ps1
```

---

## What Each Option Does

- **RemoteSigned**: Allows local scripts to run, but downloaded scripts need to be signed
- **Scope Process**: Only affects current PowerShell session
- **Scope CurrentUser**: Affects only your user account (doesn't require admin, but safer)

---

## Recommendation

**For quick setup:** Use Command Prompt (cmd.exe) - it works immediately.

**For long-term:** Use Option 2 (permanent fix) so you can use PowerShell normally in the future.

