import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { Hands } from "@mediapipe/hands";

/* ================= GAME CONFIG ================= */

const SEQUENCE = [
  "R_index",
  "L_index",
  "R_middle",
  "L_middle",
  "R_ring",
  "L_ring",
  "R_pinky",
  "L_pinky",
];

const LABEL_MAP = {
  R_index: "Right Index",
  L_index: "Left Index",
  R_middle: "Right Middle",
  L_middle: "Left Middle",
  R_ring: "Right Ring",
  L_ring: "Left Ring",
  R_pinky: "Right Pinky",
  L_pinky: "Left Pinky",
};

/* ================= APP ================= */

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const handsRef = useRef(null);
  const rafRef = useRef(null);

  const soundMapRef = useRef({
    R_index: new Audio("/sounds/R_index.mp3"),
    L_index: new Audio("/sounds/L_index.mp3"),
    R_middle: new Audio("/sounds/R_middle.mp3"),
    L_middle: new Audio("/sounds/L_middle.mp3"),
    R_ring: new Audio("/sounds/R_ring.mp3"),
    L_ring: new Audio("/sounds/L_ring.mp3"),
    R_pinky: new Audio("/sounds/R_pinky.mp3"),
    L_pinky: new Audio("/sounds/L_pinky.mp3"),
  });

  const rewardSoundRef = useRef(new Audio("/sounds/reward.mp3"));

  const [started, setStarted] = useState(false);
  const [currentGesture, setCurrentGesture] = useState("None");
  const [step, setStep] = useState(0);
  const [status, setStatus] = useState("Playing");

  const lastGestureRef = useRef(null);
  const lastTriggerRef = useRef(0);

  /* ================= MEDIAPIPE ================= */

  useEffect(() => {
    if (!started) return;
    if (handsRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 0,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults((results) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!results.multiHandLandmarks || !results.multiHandedness) return;

      results.multiHandLandmarks.forEach((landmarks, i) => {
        const raw = results.multiHandedness[i].label;
        const handLabel = raw === "Left" ? "Right" : "Left";

        const gesture = detectGesture(
          landmarks,
          handLabel,
          ctx,
          canvas
        );

        const now = Date.now();
        if (
          gesture &&
          gesture !== lastGestureRef.current &&
          now - lastTriggerRef.current > 400
        ) {
          lastGestureRef.current = gesture;
          lastTriggerRef.current = now;
          setCurrentGesture(gesture);

          const sound = soundMapRef.current[gesture];
          if (sound) {
            sound.currentTime = 0;
            sound.play();
          }

          handleGameLogic(gesture);
        }

        if (!gesture) lastGestureRef.current = null;
      });
    });

    handsRef.current = hands;

    const loop = async () => {
      if (
        webcamRef.current &&
        webcamRef.current.video.readyState === 4
      ) {
        await hands.send({ image: webcamRef.current.video });
      }
      rafRef.current = requestAnimationFrame(loop);
    };

    loop();

    return () => {
      cancelAnimationFrame(rafRef.current);
      hands.close();
      handsRef.current = null;
    };
  }, [started, step, status]);

  /* ================= GAME LOGIC ================= */

  const handleGameLogic = (gesture) => {
    if (status === "Completed") return;

    if (gesture === SEQUENCE[step]) {
      const next = step + 1;
      setStep(next);

      if (next === SEQUENCE.length) {
        setStatus("Completed");
        rewardSoundRef.current.currentTime = 0;
        rewardSoundRef.current.play();
      }
    } else {
      setStep(0);
      setStatus("Reset");
      setTimeout(() => setStatus("Playing"), 600);
    }
  };

  /* ================= UI ================= */

  if (!started) {
    return (
      <div
        style={{
          height: "100vh",
          background: "black",
          color: "white",
          padding: 40,
          overflow: "hidden",
          fontFamily: "Arial",
        }}
      >
        <h1>Kanye Gesture Game</h1>
        <p>Complete the gesture sequence to unlock the song.</p>

        <ol>
          {SEQUENCE.map((g, i) => (
            <li key={i}>{LABEL_MAP[g]}</li>
          ))}
        </ol>

        <button onClick={() => setStarted(true)}>
          Start Game
        </button>
      </div>
    );
  }

  return (
    <div
      style={{
        height: "100vh",
        background: "black",
        color: "white",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      {/* TOP BAR */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 24,
          margin: "12px 0",
          fontSize: 20,
        }}
      >
        <span style={{ fontSize: 22 }}>Status: {status}</span>
        <span style={{ fontSize: 22 }}>
          Gesture: {LABEL_MAP[currentGesture] || "None"}
        </span>

        <div style={{ display: "flex", gap: 8 }}>
          {SEQUENCE.map((_, i) => (
            <div
              key={i}
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background:
                  i < step
                    ? "lime"
                    : i === step
                    ? "orange"
                    : "#444",
              }}
            />
          ))}
        </div>
      </div>

      {/* CAMERA */}
      <div style={{ position: "relative", width: 680, height: 510 }}>
        <Webcam
          ref={webcamRef}
          mirrored
          audio={false}
          width={680}
          height={510}
          style={{ position: "absolute" }}
        />
        <canvas
          ref={canvasRef}
          width={680}
          height={510}
          style={{ position: "absolute" }}
        />
      </div>

      {status === "Completed" && (
        <div style={{ marginTop: 10, color: "lime", fontSize: 22 }}>
          🎉 SEQUENCE COMPLETED
        </div>
      )}
    </div>
  );
}

export default App;

/* ================= HELPERS ================= */

function detectGesture(landmarks, handLabel, ctx, canvas) {
  const tips = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
  const boxes = {};

  Object.entries(tips).forEach(([finger, idx]) => {
    const x = (1 - landmarks[idx].x) * canvas.width;
    const y = landmarks[idx].y * canvas.height;
    boxes[finger] = { x: x - 15, y: y - 15, w: 30, h: 30 };
  });

  Object.entries(boxes).forEach(([finger, b]) => {
    ctx.strokeStyle = "lime";
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.fillStyle = "lime";
    ctx.font = "12px Arial";
    ctx.fillText(`${handLabel} ${finger}`, b.x, b.y - 4);
  });

  for (const finger of ["index", "middle", "ring", "pinky"]) {
    if (intersects(boxes.thumb, boxes[finger])) {
      return `${handLabel[0]}_${finger}`;
    }
  }

  return null;
}

function intersects(a, b) {
  return (
    a.x < b.x + b.w &&
    a.x + a.w > b.x &&
    a.y < b.y + b.h &&
    a.y + a.h > b.y
  );
}
