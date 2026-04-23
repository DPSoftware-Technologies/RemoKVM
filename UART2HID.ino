#include "Keyboard.h"
#include <Mouse.h>
#include <MouseAbsolute.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/mutex.h"

// ─── Pin definitions ────────────────────────────────────────────────────────
#define RSTBTNPIN   5
#define PWBTNPIN    3
#define PWBLEDPIN   2
#define HDDLEDPIN   4

// ─── Protocol constants ─────────────────────────────────────────────────────
#define END_BYTE    0xFF
#define ACK_BYTE    0x06
#define NAK_BYTE    0x15
#define RESP_BYTE   0x02   // STX – start of a response payload

// ─── CMD definitions ────────────────────────────────────────────────────────
#define CMD_PING                0
#define CMD_MOUSE_MOVE_REL      1
#define CMD_MOUSE_MOVE_ABS      2
#define CMD_MOUSE_CLICK         3
#define CMD_MOUSE_HOLD          4
#define CMD_MOUSE_RELEASE       5
#define CMD_MOUSE_RELEASE_ALL   6
#define CMD_KEY_TYPE            7
#define CMD_KEY_HOLD            8
#define CMD_KEY_RELEASE         9
#define CMD_KEY_RELEASE_ALL     10
#define CMD_MOUSE_SCROLL        11
#define CMD_KB_WRITE            12
#define CMD_CTRL_ALT_DEL        13
#define CMD_CTRL_ALT_FX         14
#define CMD_CTRL_ALT_ESC        15
#define CMD_KB_COMBO            16
#define CMD_MAGIC_SYSRQ         17
#define CMD_CTRL_ALT_BKSP       18
#define CMD_SYSINFO             19
#define CMD_CPU_NOW             20
#define CMD_MEM_NOW             21
#define CMD_SPACE_NOW           22
#define CMD_NET_NOW             23
#define CMD_UPTIME_NOW          24
#define CMD_AGENT_PING          25
#define CMD_AGENT_VERSION       26
#define CMD_POWER_PRESS         27
#define CMD_POWER_HOLD          28
#define CMD_POWER_RELEASE       29
#define CMD_RESET_PRESS         30
#define CMD_GET_POWER_LED       31
#define CMD_HDD_LED_REPORT      32
#define CMD_AGENT_CMD           33

// ─── Firmware version ───────────────────────────────────────────────────────
#define AGENT_VERSION_MAJOR  1
#define AGENT_VERSION_MINOR  1
#define AGENT_VERSION_PATCH  0

// ═════════════════════════════════════════════════════════════════════════════
//
//  SHARED STATE  (written by core 1, read by core 0)
//
//  Rules:
//   • ioState       – always access while holding ioMutex
//   • evtFifo       – single-producer (core 1) / single-consumer (core 0)
//                     lock-free ring buffer; uses DMB for ordering
//   • hddLedReportEnabled – volatile bool; core 0 writes, core 1 reads,
//                           one-byte write is atomic on RP2040
//
// ═════════════════════════════════════════════════════════════════════════════

// ─── Live IO snapshot ────────────────────────────────────────────────────────
struct IOState {
  bool pwrLed;
  bool hddLed;
};
volatile IOState ioState = {false, false};
mutex_t          ioMutex;

// ─── Async event FIFO ────────────────────────────────────────────────────────
// Event types sent from core 1 → core 0
#define EVT_HDD_LED  0xE1   // HDD activity LED changed  (value = 0/1)
#define EVT_PWR_LED  0xE2   // Power LED changed          (value = 0/1)

#define FIFO_DEPTH   16

struct Event {
  uint8_t type;
  uint8_t value;
};

volatile Event  evtFifo[FIFO_DEPTH];
volatile uint8_t evtHead = 0;  // advanced by core 1 (producer)
volatile uint8_t evtTail = 0;  // advanced by core 0 (consumer)

// Push an event (core 1 only). Silently drops if the FIFO is full.
inline void evtPush(uint8_t type, uint8_t value) {
  uint8_t next = (evtHead + 1) % FIFO_DEPTH;
  if (next == evtTail) return;          // full – drop
  evtFifo[evtHead].type  = type;
  evtFifo[evtHead].value = value;
  __dmb();                              // ensure write is visible before head advance
  evtHead = next;
}
// Pop an event (core 0 only). Returns false when empty. [cite: 16]
inline bool evtPop(Event &out) {
  if (evtTail == evtHead) return false;

  // Copy members individually to safely handle volatile memory
  out.type = evtFifo[evtTail].type;
  out.value = evtFifo[evtTail].value;

  __dmb(); // ensure data is read before tail advances
  evtTail = (evtTail + 1) % FIFO_DEPTH;
  return true;
}
// HDD-LED report enable flag – core 0 writes, core 1 reads
volatile bool hddLedReportEnabled = true;

// ═════════════════════════════════════════════════════════════════════════════
//  CORE 1  –  IO polling task
//
//  Responsibilities:
//    • Debounce and sample HDDLEDPIN and PWBLEDPIN at ~2 kHz
//    • Update ioState (mutex-protected)
//    • Push change events into evtFifo
//
//  Does NOT touch: Serial, Serial1, HID, or any output pin
// ═════════════════════════════════════════════════════════════════════════════

void core1_io_task() {
  const uint32_t DEBOUNCE_MS = 5;

  bool     lastHdd = false, lastPwr = false;
  uint32_t hddStableAt = 0, pwrStableAt = 0;
  bool     hddPending  = false, pwrPending  = false;
  bool     hddPendVal  = false, pwrPendVal  = false;

  while (true) {
    uint32_t now = to_ms_since_boot(get_absolute_time());

    bool hddRaw = (digitalRead(HDDLEDPIN) == HIGH);
    bool pwrRaw = (digitalRead(PWBLEDPIN) == HIGH);

    // ── HDD LED debounce ─────────────────────────────────────────────────
    if (hddRaw != lastHdd) {
      if (!hddPending || hddRaw != hddPendVal) {
        // New edge – start / restart the debounce timer
        hddPending  = true;
        hddPendVal  = hddRaw;
        hddStableAt = now + DEBOUNCE_MS;
      }
    }
    if (hddPending && (int32_t)(now - hddStableAt) >= 0) {
      // Signal has been stable for DEBOUNCE_MS
      lastHdd    = hddPendVal;
      hddPending = false;

      mutex_enter_blocking(&ioMutex);
      ioState.hddLed = lastHdd;
      mutex_exit(&ioMutex);

      if (hddLedReportEnabled) {
        evtPush(EVT_HDD_LED, lastHdd ? 1 : 0);
      }
    }

    // ── Power LED debounce ───────────────────────────────────────────────
    if (pwrRaw != lastPwr) {
      if (!pwrPending || pwrRaw != pwrPendVal) {
        pwrPending  = true;
        pwrPendVal  = pwrRaw;
        pwrStableAt = now + DEBOUNCE_MS;
      }
    }
    if (pwrPending && (int32_t)(now - pwrStableAt) >= 0) {
      lastPwr    = pwrPendVal;
      pwrPending = false;

      mutex_enter_blocking(&ioMutex);
      ioState.pwrLed = lastPwr;
      mutex_exit(&ioMutex);

      // Always emit power-LED events (useful for detecting host on/off)
      evtPush(EVT_PWR_LED, lastPwr ? 1 : 0);
    }

    sleep_us(500); // ~2 kHz polling – fast enough, very low overhead
  }
}

// ═════════════════════════════════════════════════════════════════════════════
//  CORE 0  –  Serial1 helpers
// ═════════════════════════════════════════════════════════════════════════════

void sendACK()  { Serial1.write(ACK_BYTE); }
void sendNAK()  { Serial1.write(NAK_BYTE); }

void sendResponse(const uint8_t *data, uint8_t len) {
  Serial1.write(RESP_BYTE);
  Serial1.write(len);
  Serial1.write(data, len);
}

// Drain any pending IO events from the FIFO and push them over Serial1.
// Called opportunistically inside blocking waits so events are never held up.
inline void drainEvents() {
  Event e;
  while (evtPop(e)) {
    uint8_t buf[2] = {e.type, e.value};
    sendResponse(buf, 2);
  }
}

// Blocking byte read from Serial1; drains events while waiting.
int readByteTimeout(unsigned long timeout_ms = 500) {
  unsigned long start = millis();
  while (!Serial1.available()) {
    if (millis() - start > timeout_ms) return -1;
    drainEvents();
  }
  return Serial1.read();
}

bool readBytes(uint8_t *buf, uint8_t n, unsigned long timeout_ms = 500) {
  for (uint8_t i = 0; i < n; i++) {
    int b = readByteTimeout(timeout_ms);
    if (b < 0) return false;
    buf[i] = (uint8_t)b;
  }
  return true;
}

int readUntilEnd(uint8_t *buf, uint8_t maxLen, unsigned long timeout_ms = 2000) {
  uint8_t count = 0;
  while (count < maxLen) {
    int b = readByteTimeout(timeout_ms);
    if (b < 0)         return -1;
    if (b == END_BYTE) return count;
    buf[count++] = (uint8_t)b;
  }
  return -1; // overflow
}

// ─── Misc helpers ────────────────────────────────────────────────────────────
int mapMouseButton(uint8_t btn) {
  switch (btn) {
    case 0: return MOUSE_LEFT;
    case 1: return MOUSE_MIDDLE;
    case 2: return MOUSE_RIGHT;
    default: return MOUSE_LEFT;
  }
}

char mapSysRqKey(uint8_t k) {
  const char keys[] = {'r', 'e', 'i', 's', 'u', 'b', 'o'};
  return (k < 7) ? keys[k] : 'b';
}

// ═════════════════════════════════════════════════════════════════════════════
//  CORE 0  –  Command handlers  (HID / Serial / GPIO output only)
// ═════════════════════════════════════════════════════════════════════════════

void handlePing() {
  uint8_t r[1] = {0x50};
  sendResponse(r, 1);
}

void handleMouseMoveRel() {
  uint8_t buf[2];
  if (!readBytes(buf, 2)) { sendNAK(); return; }
  Mouse.move((int8_t)buf[0], (int8_t)buf[1], 0);
  sendACK();
}

void handleMouseMoveAbs() {
  uint8_t buf[4];
  if (!readBytes(buf, 4)) { sendNAK(); return; }
  int16_t x = (int16_t)((buf[0] << 8) | buf[1]);
  int16_t y = (int16_t)((buf[2] << 8) | buf[3]);
  MouseAbsolute.move(x, y, 0);
  sendACK();
}

void handleMouseClick() {
  int btn = readByteTimeout();
  if (btn < 0) { sendNAK(); return; }
  Mouse.click(mapMouseButton(btn));
  sendACK();
}

void handleMouseHold() {
  int btn = readByteTimeout();
  if (btn < 0) { sendNAK(); return; }
  Mouse.press(mapMouseButton(btn));
  sendACK();
}

void handleMouseRelease() {
  int btn = readByteTimeout();
  if (btn < 0) { sendNAK(); return; }
  Mouse.release(mapMouseButton(btn));
  sendACK();
}

void handleMouseReleaseAll() {
  Mouse.release(MOUSE_LEFT);
  Mouse.release(MOUSE_MIDDLE);
  Mouse.release(MOUSE_RIGHT);
  sendACK();
}

void handleKeyType() {
  int key = readByteTimeout();
  if (key < 0) { sendNAK(); return; }
  Keyboard.press(key);
  delay(10);
  Keyboard.release(key);
  sendACK();
}

void handleKeyHold() {
  int key = readByteTimeout();
  if (key < 0) { sendNAK(); return; }
  Keyboard.press(key);
  sendACK();
}

void handleKeyRelease() {
  int key = readByteTimeout();
  if (key < 0) { sendNAK(); return; }
  Keyboard.release(key);
  sendACK();
}

void handleKeyReleaseAll() {
  Keyboard.releaseAll();
  sendACK();
}

void handleMouseScroll() {
  int delta = readByteTimeout();
  if (delta < 0) { sendNAK(); return; }
  Mouse.move(0, 0, (int8_t)delta);
  sendACK();
}

void handleKbWrite() {
  uint8_t buf[256];
  int len = readUntilEnd(buf, 255);
  if (len < 0) { sendNAK(); return; }
  for (int i = 0; i < len; i++) {
    Keyboard.press(buf[i]);
    delay(5);
    Keyboard.release(buf[i]);
    delay(5);
  }
  sendACK();
}

void handleCtrlAltDel() {
  Keyboard.press(KEY_LEFT_CTRL);
  Keyboard.press(KEY_LEFT_ALT);
  Keyboard.press(KEY_DELETE);
  delay(100);
  Keyboard.releaseAll();
  sendACK();
}

void handleCtrlAltFx() {
  int fx = readByteTimeout();
  if (fx < 1 || fx > 12) { sendNAK(); return; }
  const uint8_t fkeys[12] = {
    KEY_F1, KEY_F2, KEY_F3,  KEY_F4,  KEY_F5,  KEY_F6,
    KEY_F7, KEY_F8, KEY_F9, KEY_F10, KEY_F11, KEY_F12
  };
  Keyboard.press(KEY_LEFT_CTRL);
  Keyboard.press(KEY_LEFT_ALT);
  Keyboard.press(fkeys[fx - 1]);
  delay(100);
  Keyboard.releaseAll();
  sendACK();
}

void handleCtrlAltEsc() {
  Keyboard.press(KEY_LEFT_CTRL);
  Keyboard.press(KEY_LEFT_ALT);
  Keyboard.press(KEY_ESC);
  delay(100);
  Keyboard.releaseAll();
  sendACK();
}

void handleKbCombo() {
  uint8_t buf[2];
  if (!readBytes(buf, 2)) { sendNAK(); return; }
  Keyboard.press(buf[0]);
  Keyboard.press(buf[1]);
  delay(50);
  Keyboard.releaseAll();
  sendACK();
}

void handleMagicSysRq() {
  int k = readByteTimeout();
  if (k < 0 || k > 6) { sendNAK(); return; }
  Keyboard.press(KEY_LEFT_ALT);
  Keyboard.press(KEY_PRINT_SCREEN);
  delay(50);
  Keyboard.press(mapSysRqKey(k));
  delay(100);
  Keyboard.releaseAll();
  sendACK();
}

void handleCtrlAltBackspace() {
  Keyboard.press(KEY_LEFT_CTRL);
  Keyboard.press(KEY_LEFT_ALT);
  Keyboard.press(KEY_BACKSPACE);
  delay(100);
  Keyboard.releaseAll();
  sendACK();
}

void forwardToAgent(uint8_t cmd) {
  Serial.write(RESP_BYTE);
  Serial.write(cmd);
  Serial.write(END_BYTE);

  uint8_t respBuf[512];
  unsigned long start = millis();
  uint8_t idx = 0;
  bool gotResp = false;

  while (millis() - start < 3000 && idx < sizeof(respBuf) - 1) {
    if (Serial.available()) {
      uint8_t b = Serial.read();
      if (b == END_BYTE) { gotResp = true; break; }
      respBuf[idx++] = b;
    }
    drainEvents(); // keep events flowing while waiting for CDC response
  }

  if (gotResp && idx > 0) sendResponse(respBuf, idx);
  else                     sendNAK();
}

void handlePowerPress() {
  digitalWrite(PWBTNPIN, HIGH);
  delay(200);
  digitalWrite(PWBTNPIN, LOW);
  sendACK();
}

void handlePowerHold()    { digitalWrite(PWBTNPIN, HIGH); sendACK(); }
void handlePowerRelease() { digitalWrite(PWBTNPIN, LOW);  sendACK(); }

void handleResetPress() {
  digitalWrite(RSTBTNPIN, HIGH);
  delay(200);
  digitalWrite(RSTBTNPIN, LOW);
  sendACK();
}

// CMD 31: Read power LED from the shared snapshot – no pin read on core 0
void handleGetPowerLed() {
  mutex_enter_blocking(&ioMutex);
  bool state = ioState.pwrLed;
  mutex_exit(&ioMutex);

  // Cast to uint8_t to prevent narrowing conversion warnings
  uint8_t resp[1] = {(uint8_t)(state ? 1 : 0)};
  sendResponse(resp, 1);
}

void handleHddLedReport() {
  int ena = readByteTimeout();
  if (ena < 0) { sendNAK(); return; }
  hddLedReportEnabled = (ena == 1); // volatile – core 1 picks this up
  sendACK();
}

void handleAgentCmd() {
  uint8_t payload[256];
  int len = readUntilEnd(payload, 255, 3000);
  if (len < 0) { sendNAK(); return; }

  Serial.write(RESP_BYTE);
  Serial.write((uint8_t)0x21);
  if (len > 0) Serial.write(payload, len);
  Serial.write(END_BYTE);

  uint8_t respBuf[512];
  unsigned long start = millis();
  uint8_t idx = 0;
  bool gotResp = false;

  while (millis() - start < 5000 && idx < sizeof(respBuf) - 1) {
    if (Serial.available()) {
      uint8_t b = Serial.read();
      if (b == END_BYTE) { gotResp = true; break; }
      if (b == NAK_BYTE) { sendNAK(); return; }
      respBuf[idx++] = b;
    }
    drainEvents();
  }

  if (gotResp && idx > 0) sendResponse(respBuf, idx);
  else                     sendNAK();
}

// ═════════════════════════════════════════════════════════════════════════════
//  setup1()  –  CORE 1 entry point (Arduino-Pico framework)
//               Pins are already configured by the time this runs.
// ═════════════════════════════════════════════════════════════════════════════
void setup1() {
  core1_io_task(); // infinite loop – never returns
}

// ═════════════════════════════════════════════════════════════════════════════
//  setup()  –  CORE 0
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
  mutex_init(&ioMutex); // must be done before core 1 starts

  Serial.begin(115200);   // USB CDC → Pi 5 agent daemon
  Serial1.begin(115200);  // Physical UART ↔ Pi 5 controller

  // Output pins
  pinMode(PWBTNPIN,  OUTPUT);
  pinMode(RSTBTNPIN, OUTPUT);
  digitalWrite(PWBTNPIN,  LOW);
  digitalWrite(RSTBTNPIN, LOW);

  // Input pins (read by core 1 only after this point)
  pinMode(PWBLEDPIN, INPUT);
  pinMode(HDDLEDPIN, INPUT);

  Mouse.begin();
  MouseAbsolute.begin();
  Keyboard.begin();

  // The Arduino-Pico framework launches core 1 (calls setup1()) automatically
  // after setup() returns.
}

// ═════════════════════════════════════════════════════════════════════════════
//  loop()  –  CORE 0  –  command dispatch + async event drain
// ═════════════════════════════════════════════════════════════════════════════
void loop() {
  // ── Step 1: flush any IO events queued by core 1 ─────────────────────────
  drainEvents();

  // ── Step 2: handle one Serial1 command if available ──────────────────────
  if (!Serial1.available()) return;

  uint8_t cmd = Serial1.read();

  switch (cmd) {
    case CMD_PING:               handlePing();              break;
    case CMD_MOUSE_MOVE_REL:     handleMouseMoveRel();      break;
    case CMD_MOUSE_MOVE_ABS:     handleMouseMoveAbs();      break;
    case CMD_MOUSE_CLICK:        handleMouseClick();        break;
    case CMD_MOUSE_HOLD:         handleMouseHold();         break;
    case CMD_MOUSE_RELEASE:      handleMouseRelease();      break;
    case CMD_MOUSE_RELEASE_ALL:  handleMouseReleaseAll();   break;
    case CMD_KEY_TYPE:           handleKeyType();           break;
    case CMD_KEY_HOLD:           handleKeyHold();           break;
    case CMD_KEY_RELEASE:        handleKeyRelease();        break;
    case CMD_KEY_RELEASE_ALL:    handleKeyReleaseAll();     break;
    case CMD_MOUSE_SCROLL:       handleMouseScroll();       break;
    case CMD_KB_WRITE:           handleKbWrite();           break;
    case CMD_CTRL_ALT_DEL:       handleCtrlAltDel();        break;
    case CMD_CTRL_ALT_FX:        handleCtrlAltFx();         break;
    case CMD_CTRL_ALT_ESC:       handleCtrlAltEsc();        break;
    case CMD_KB_COMBO:           handleKbCombo();           break;
    case CMD_MAGIC_SYSRQ:        handleMagicSysRq();        break;
    case CMD_CTRL_ALT_BKSP:      handleCtrlAltBackspace();  break;
    case CMD_SYSINFO:
    case CMD_CPU_NOW:
    case CMD_MEM_NOW:
    case CMD_SPACE_NOW:
    case CMD_NET_NOW:
    case CMD_UPTIME_NOW:         forwardToAgent(cmd);       break;
    case CMD_AGENT_PING: {
      uint8_t r[1] = {0x50};
      sendResponse(r, 1);
      break;
    }
    case CMD_AGENT_VERSION: {
      uint8_t r[3] = {AGENT_VERSION_MAJOR, AGENT_VERSION_MINOR, AGENT_VERSION_PATCH};
      sendResponse(r, 3);
      break;
    }
    case CMD_POWER_PRESS:        handlePowerPress();        break;
    case CMD_POWER_HOLD:         handlePowerHold();         break;
    case CMD_POWER_RELEASE:      handlePowerRelease();      break;
    case CMD_RESET_PRESS:        handleResetPress();        break;
    case CMD_GET_POWER_LED:      handleGetPowerLed();       break;
    case CMD_HDD_LED_REPORT:     handleHddLedReport();      break;
    case CMD_AGENT_CMD:          handleAgentCmd();          break;
    default:                     sendNAK();                 break;
  }
}

// loop1() is intentionally absent — core1_io_task() never returns from setup1()
