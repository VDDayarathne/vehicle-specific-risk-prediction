# 🔍 KaduGuard Registration Debugging Guide

## Problem: Registration is Not Working

Follow these steps **in order** to identify the exact error:

---

## ✅ Step 1: Check if Backend is Running

Open PowerShell and run:

```powershell
cd "C:\Users\ASUS\Downloads\datasets\New folder"
docker compose ps
```

**Expected output:**
```
NAME      IMAGE             STATUS
backend   kaduguard-backend Up 2 minutes
db        postgres:15       Up 2 minutes
```

**If services are NOT running**, start them:
```powershell
docker compose up -d
```

**If status shows "Exited"**, check logs:
```powershell
docker compose logs backend
docker compose logs db
```

---

## ✅ Step 2: View Backend Logs (Real-time)

This shows exactly what happens when registration is attempted:

```powershell
docker compose logs backend -f
```

**Keep this running** while you try to register in the mobile app.

**Look for:**
- ✅ If you see `POST /api/auth/register` → Request reached backend
- ❌ If you see connection errors → Backend isn't accessible
- ❌ If you see database errors → Database connection failed
- ❌ If you see validation errors → Data format is wrong

---

## ✅ Step 3: Check Android Logs (Logcat)

Open Android Studio and view real-time logs:

1. **Open Android Studio** with the project
2. **Run → Edit Configurations** → Select your emulator/device
3. **Run the app** (Shift + F10 or green play button)
4. **View → Tool Windows → Logcat** (or Alt + 6)

**Filter for registration logs:**
- Search for: `AuthViewModel`
- You'll see logs like:
  ```
  D/AuthViewModel: Starting Register with email: test@example.com
  D/AuthViewModel: Attempting registration with vehicle_type: car
  E/AuthViewModel: HTTP Error: Server Error (400): ...
  ```

---

## ✅ Step 4: Test Backend API Directly

Test the registration endpoint **without the mobile app** to isolate the problem:

```powershell
$body = @{
    email = "testuser@example.com"
    password = "password123"
    phone = "+94123456789"
    vehicle_type = "car"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/auth/register" `
  -Method Post `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body | Select-Object StatusCode, Content
```

**Expected response (StatusCode 200):**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ..."
}
```

**Common errors:**

| Status | Error | Solution |
|--------|-------|----------|
| 409 | "User with this email already exists" | Try a different email |
| 422 | "Validation error" | Check field names match (`vehicle_type`, not `vehicleType`) |
| 500 | "Internal Server Error" | Check backend logs with `docker compose logs backend` |
| Connection refused | Backend not running | Run `docker compose up -d` |

---

## ✅ Step 5: Verify Network Connectivity

### For Emulator:
```powershell
# Test if emulator can reach backend on host
adb shell ping 10.0.2.2
```

If ping fails, the emulator can't reach your PC. In that case, update `ApiBaseUrl` in:
```
android/app/src/main/res/values/strings.xml
```

### For Physical Device:
```bash
# Find your PC's IP address (use this instead of 10.0.2.2)
ipconfig
```

Look for IPv4 address like `192.168.x.x` and update:
```
android/app/src/main/res/values/strings.xml
```

---

## 🐛 Common Issues and Solutions

### Issue 1: "Cannot connect to backend"
- **Cause:** Emulator can't reach `10.0.2.2:8000`
- **Fix:** 
  - For emulator: Ensure backend is running on Windows host at `localhost:8000`
  - For physical device: Use your PC's IP instead (e.g., `http://192.168.1.100:8000`)

### Issue 2: "Email already exists"
- **Cause:** You're using the same email twice
- **Fix:** Try a new email like `test_{{timestamp}}@example.com`

### Issue 3: "Server Error (422): Validation error"
- **Cause:** Wrong field names or types
- **Fix:** Check that you're sending `vehicle_type` (snake_case, not camelCase)
- **Check:** Backend accepts: `email`, `password`, `phone` (optional), `vehicle_type`

### Issue 4: "Connection timeout"
- **Cause:** Backend is running but not responding quickly
- **Fix:** Check backend logs for slow queries or errors

### Issue 5: Error message is blank/generic
- **Cause:** Error details not being captured
- **Fix:** The enhanced AuthViewModel now shows detailed errors. Rebuild:
  ```bash
  cd android
  .\gradlew.bat clean assembleDebug
  ```

---

## 📋 Complete Debugging Checklist

### Before Testing:
- [ ] Run `docker compose ps` and confirm **both services are UP**
- [ ] Database file exists and is accessible
- [ ] No other services on port 8000 (check with `netstat -ano | findstr 8000`)

### During Testing (Mobile App):
- [ ] Open **Logcat** before clicking Register
- [ ] Scroll to see **all AuthViewModel logs**
- [ ] Check **exact error message** displayed on screen

### After Failure:
- [ ] Copy **error message** from screen
- [ ] Copy **Logcat logs** (right-click → Copy)
- [ ] Check **backend logs** with `docker compose logs backend`
- [ ] Test **manually with curl** command above

---

## 📞 Share These Details to Debug

When reporting the issue, provide:

1. **Screenshot of the error** on mobile app
2. **Logcat output** with "AuthViewModel" logs
3. **Backend logs** from `docker compose logs backend`
4. **Result of manual API test** with curl
5. **Docker status** from `docker compose ps`

---

## 🚀 Quick Test Sequence

Run this sequence to diagnose the issue:

```powershell
# Terminal 1: Start backend and monitor logs
cd "C:\Users\ASUS\Downloads\datasets\New folder"
docker compose up -d
docker compose logs backend -f

# Terminal 2: Test API manually while backend logs show above
$body = @{
    email = "quicktest@example.com"
    password = "test123"
    phone = "+94123456789"
    vehicle_type = "car"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/auth/register" `
  -Method Post `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body
```

Then **try registration in the mobile app** while watching both Logcat and backend logs.

---

**Questions?** Share the outputs from steps 2, 3, and 4 above, and I can identify the exact issue! 🎯
