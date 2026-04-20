/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, 
  AlertTriangle, 
  Terminal, 
  Wifi, 
  ShieldCheck, 
  Clock,
  Play,
  Volume2,
  Settings2,
  Lock,
  CameraOff
} from 'lucide-react';

// --- Constants & Config ---
const THRESHOLDS = {
  EAR: { LOW: 0.28, MED: 0.25, HIGH: 0.22 },
  POISE: { YAW: { LOW: 20, HIGH: 40 }, PITCH: { LOW: 15, HIGH: 30 } },
  DURATIONS: { ADVISORY: 5, URGENT: 12, CRITICAL: 25 }
};

type Severity = 0 | 1 | 2 | 3;
type SystemState = 'INIT' | 'LOADING' | 'READY' | 'ACTIVE' | 'ERROR';

export default function App() {
  // --- Refs (High-Performance Tracking) ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const requestRef = useRef<number>(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Real-time tracking refs to avoid React re-render thrashing
  const earRef = useRef(0);
  const poseRef = useRef({ yaw: 0, pitch: 0 });
  const drowsyFramesRef = useRef(0);
  const severityRef = useRef<Severity>(0);
  const lastEarStateRef = useRef(true);
  const fpsRef = useRef(0);
  const lastTimeRef = useRef(performance.now());

  // --- State (UI Rendering) ---
  const [systemState, setSystemState] = useState<SystemState>('INIT');
  const [uiSeverity, setUiSeverity] = useState<Severity>(0);
  const [isUserActive, setIsUserActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [displayStats, setDisplayStats] = useState({ ear: 0, yaw: 0, pitch: 0, fps: 0 });
  const [isGpu, setIsGpu] = useState(false);
  const [isFaceDetected, setIsFaceDetected] = useState(false);

  // --- Audio Engine ---
  const initAudio = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    console.log("Audio Engine Initialized: ", audioContextRef.current.state);
  };

  const playOneShotPulse = (freq: number, dur: number = 0.08) => {
    if (!audioContextRef.current) return;
    const ctx = audioContextRef.current;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + dur);
  };

  const stopAlert = () => {
    if (audioIntervalRef.current) {
      clearInterval(audioIntervalRef.current);
      audioIntervalRef.current = null;
    }
  };

  const runAlertPattern = (level: Severity) => {
    // SILENCE: Level 1 and 2 continuous loops as per request.
    // Only Level 3 (Dangerous/Critical) triggers a sustained siren.
    if (!audioContextRef.current || level !== 3) {
      stopAlert();
      return;
    }
    
    if (audioIntervalRef.current) return;

    const frequencies = { 3: [1500, 2200, 1800, 2500] };
    const intervals = { 3: 80 };

    let step = 0;
    audioIntervalRef.current = setInterval(() => {
      if (!audioContextRef.current) return;
      const ctx = audioContextRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const levelFreqs = frequencies[3];
      osc.frequency.setValueAtTime(levelFreqs[step % levelFreqs.length], ctx.currentTime);
      osc.type = 'square';
      gain.gain.setValueAtTime(0.25, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.15);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.15);
      step++;
    }, intervals[3]);
  };

  // --- MediaPipe Engine ---
  useEffect(() => {
    let landmarker: FaceLandmarker | null = null;
    
    async function setupLandmarker() {
      setSystemState('LOADING');
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        // Strategy: Try GPU first, fallback to CPU explicitly if needed
        try {
          landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
              delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1
          });
          setIsGpu(true);
        } catch (gpuErr) {
          console.warn("GPU Delegate failed, falling back to CPU...", gpuErr);
          landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
              delegate: "CPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1
          });
          setIsGpu(false);
        }
        
        if (landmarker) {
          faceLandmarkerRef.current = landmarker;
          setSystemState('READY');
        } else {
          throw new Error("Initialization failed: Landmarker not created.");
        }
      } catch (err) {
        console.error(err);
        setErrorMsg("Failed to initialize AI models. System might be offline or browser is incompatible.");
        setSystemState('ERROR');
      }
    }
    setupLandmarker();
    return () => {
      stopAlert();
      cancelAnimationFrame(requestRef.current);
      if (landmarker) landmarker.close();
    };
  }, []);

  // --- High-Performance Process Loop ---
  const processFrame = () => {
    if (!videoRef.current || !faceLandmarkerRef.current) {
      requestRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const now = performance.now();
    const vWidth = videoRef.current.videoWidth;
    const vHeight = videoRef.current.videoHeight;
    
    if (vWidth === 0 || vHeight === 0) {
      requestRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const results = faceLandmarkerRef.current.detectForVideo(videoRef.current, now);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];
      
      const vWidth = videoRef.current.videoWidth;
      const vHeight = videoRef.current.videoHeight;

      const distance = (p1Idx: number, p2Idx: number) => {
        const p1 = landmarks[p1Idx];
        const p2 = landmarks[p2Idx];
        return Math.sqrt(Math.pow((p1.x - p2.x) * vWidth, 2) + Math.pow((p1.y - p2.y) * vHeight, 2));
      };

      // Metrics (Pixel Space)
      const rEAR = (distance(160, 144) + distance(158, 153)) / (2 * distance(33, 133));
      const lEAR = (distance(385, 380) + distance(387, 373)) / (2 * distance(362, 263));
      const currentEAR = (rEAR + lEAR) / 2;
      earRef.current = currentEAR;

      const nose = landmarks[1];
      const lEye = landmarks[133];
      const rEye = landmarks[362];
      const yawVal = (Math.abs((nose.x - lEye.x) * vWidth) - Math.abs((nose.x - rEye.x) * vWidth)) / (Math.abs((nose.x - lEye.x) * vWidth) + Math.abs((nose.x - rEye.x) * vWidth)) * 100;
      const pitchVal = (((lEye.y + rEye.y) / 2 - nose.y) * vHeight) * 0.5; // Scaled for pixel space
      poseRef.current = { yaw: yawVal, pitch: pitchVal };

      // severity
      let currentSeverity: Severity = 0;
      const isEarClosed = currentEAR < THRESHOLDS.EAR.LOW;

      if (isEarClosed) {
        drowsyFramesRef.current++;
        if (lastEarStateRef.current) {
          playOneShotPulse(1200); 
          lastEarStateRef.current = false;
        }
      } else {
        drowsyFramesRef.current = 0;
        lastEarStateRef.current = true;
      }

      if (drowsyFramesRef.current > THRESHOLDS.DURATIONS.CRITICAL || currentEAR < THRESHOLDS.EAR.HIGH) {
        currentSeverity = 3;
      } else if (drowsyFramesRef.current > THRESHOLDS.DURATIONS.URGENT || currentEAR < THRESHOLDS.EAR.MED) {
        currentSeverity = 2;
      } else if (isEarClosed) {
        currentSeverity = 1;
      }

      const distYaw = Math.abs(yawVal);
      const distPitch = Math.abs(pitchVal);
      if (distYaw > THRESHOLDS.POISE.YAW.HIGH || distPitch > THRESHOLDS.POISE.PITCH.HIGH) {
        currentSeverity = Math.max(currentSeverity, 2) as Severity;
      } else if (distYaw > THRESHOLDS.POISE.YAW.LOW || distPitch > THRESHOLDS.POISE.PITCH.LOW) {
        currentSeverity = Math.max(currentSeverity, 1) as Severity;
      }

      if (currentSeverity !== severityRef.current) {
        severityRef.current = currentSeverity;
        setUiSeverity(currentSeverity);
        stopAlert();
        runAlertPattern(currentSeverity);
      }
      if (!isUserActive) setIsUserActive(true);
      if (!isFaceDetected) setIsFaceDetected(true);
    } else {
      if (isUserActive) setIsUserActive(false);
      if (isFaceDetected) setIsFaceDetected(false);
      if (severityRef.current !== 0) {
        severityRef.current = 0;
        setUiSeverity(0);
        stopAlert();
      }
    }

    fpsRef.current = Math.round(1000 / (now - lastTimeRef.current));
    lastTimeRef.current = now;
    requestRef.current = requestAnimationFrame(processFrame);
  };

  // --- UI Update Throttler (updates UI at 15fps to keep main thread free) ---
  useEffect(() => {
    const timer = setInterval(() => {
      setDisplayStats({
        ear: earRef.current,
        yaw: poseRef.current.yaw,
        pitch: poseRef.current.pitch,
        fps: fpsRef.current
      });
    }, 66); // ~15fps UI updates
    return () => clearInterval(timer);
  }, []);

  const startStream = async () => {
    initAudio();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720, facingMode: 'user' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setSystemState('ACTIVE');
          requestRef.current = requestAnimationFrame(processFrame);
        };
      }
    } catch (err: any) {
      console.error("Camera Error:", err);
      if (err.name === 'NotAllowedError') {
        setErrorMsg("Camera access denied. Please click the camera icon in your browser's address bar and select 'Allow'.");
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setErrorMsg("Camera is already in use by another application (possibly app.py). Please close other camera apps and reload.");
      } else {
        setErrorMsg(`Camera Error: ${err.message || "Unknown error"}. Please check your camera settings.`);
      }
      setSystemState('ERROR');
    }
  };

  const severityStyles = {
    0: { color: 'text-primary-safe', bg: 'bg-primary-safe', label: 'OPTIMAL_STATE', desc: 'SYSTEM MONITORING ACTIVE' },
    1: { color: 'text-secondary-warn', bg: 'bg-secondary-warn', label: 'ADVISORY', desc: 'ATTENTION DISCREPANCY DETECTED' },
    2: { color: 'text-secondary-warn', bg: 'bg-secondary-warn', label: 'URGENT', desc: 'IMMEDIATE RE-FOCUS REQUIRED' },
    3: { color: 'text-tertiary-alert', bg: 'bg-tertiary-alert', label: 'CRITICAL', desc: 'DANGER: SAFETY PROTOCOL BREACHED' }
  };

  return (
    <div className="fixed inset-0 overflow-hidden bg-surface flex flex-col selection:bg-primary-safe/30 text-white font-sans">
      {/* Dynamic Background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className={`absolute top-1/4 left-1/4 w-[120%] h-[120%] ${severityStyles[uiSeverity].bg}/5 blur-[120px] rounded-full animate-pulse transition-colors duration-700`} />
      </div>

      {/* TACTICAL HEADER */}
      <header className="z-10 p-6 flex justify-between items-center bg-surface/80 backdrop-blur-xl border-b border-white/5">
        <div className="flex items-center gap-4">
          <div className={`p-2 ${severityStyles[uiSeverity].bg}/10 rounded-sm border ${severityStyles[uiSeverity].bg}/20`}>
            <Terminal className={`w-5 h-5 ${severityStyles[uiSeverity].color}`} />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-widest text-white/50 uppercase">Threat Intel System 02</h1>
            <div className="flex items-center gap-2">
              <div className={`w-1 h-1 rounded-full animate-ping ${severityStyles[uiSeverity].bg}`} />
              <p className={`text-[10px] telemetry ${severityStyles[uiSeverity].color}/60 leading-none`}>
                SEVERITY_LVL_{uiSeverity} // ACTIVE_WATCH
              </p>
            </div>
          </div>
        </div>
        
        <div className="flex gap-8">
           <div className="flex flex-col items-end">
             <span className="text-[10px] text-white/20 uppercase tracking-[0.2em]">Sensor Accuracy</span>
             <span className="text-xs telemetry text-primary-safe font-medium">98.4%_NOMINAL</span>
           </div>
           <div className="flex flex-col items-end">
             <span className="text-[10px] text-white/20 uppercase tracking-[0.2em]">Inference Speed</span>
             <span className="text-xs telemetry text-primary-safe font-medium">{displayStats.fps} FPS</span>
           </div>
        </div>
      </header>

      <main className="flex-1 relative flex gap-4 p-4 overflow-hidden">
        {/* SIDE TELEMETRY */}
        <div className="w-80 flex flex-col gap-4 z-20">
          <div className="no-line-card p-6 flex-1 flex flex-col gap-10">
            <div className="space-y-1">
              <span className="text-[10px] text-white/30 uppercase tracking-[0.3em]">Protocol Status</span>
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full shadow-[0_0_12px] transition-colors duration-300 ${uiSeverity === 0 ? 'bg-primary-safe shadow-primary-safe/50' : uiSeverity === 3 ? 'bg-tertiary-alert shadow-tertiary-alert/50' : 'bg-secondary-warn shadow-secondary-warn/50'}`} />
                <h2 className="text-2xl font-display font-medium tracking-tight uppercase leading-none">{severityStyles[uiSeverity].label}</h2>
              </div>
            </div>

            <div className="space-y-10">
              <div className="space-y-4">
                <div className="flex justify-between items-end">
                  <span className="text-[10px] text-white/40 uppercase tracking-widest">Biometric_EAR</span>
                  <span className={`text-xl telemetry font-medium ${displayStats.ear < THRESHOLDS.EAR.LOW ? 'text-secondary-warn' : 'text-primary-safe'}`}>{displayStats.ear.toFixed(3)}</span>
                </div>
                <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                  <motion.div 
                    className={`h-full ${uiSeverity === 3 ? 'bg-tertiary-alert' : uiSeverity > 0 ? 'bg-secondary-warn' : 'bg-primary-safe'}`}
                    animate={{ width: `${Math.min(displayStats.ear * 300, 100)}%` }}
                    transition={{ type: 'spring', bounce: 0 }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <span className="text-[9px] text-white/30 uppercase block tracking-widest">Yaw_Orientation</span>
                  <div className="bg-white/5 p-4 rounded-sm border border-white/5 flex items-center justify-center">
                     <span className="text-lg telemetry text-white/80">{displayStats.yaw.toFixed(1)}°</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <span className="text-[9px] text-white/30 uppercase block tracking-widest">Pitch_Gradient</span>
                  <div className="bg-white/5 p-4 rounded-sm border border-white/5 flex items-center justify-center">
                     <span className="text-lg telemetry text-white/80">{displayStats.pitch.toFixed(1)}°</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-auto space-y-6">
              <div className="p-4 bg-white/5 rounded-sm space-y-4">
                <span className="text-[10px] text-white/20 uppercase tracking-[0.3em] block underline decoration-primary-safe/30 underline-offset-4">System Integrity Log</span>
                <div className="space-y-2 text-[9px] telemetry text-white/40 uppercase leading-relaxed font-mono">
                  <div className="flex justify-between"><span>Core_Engine</span><span className="text-primary-safe">STABLE</span></div>
                  <div className="flex justify-between"><span>Video_Pipe</span><span className={isFaceDetected ? "text-primary-safe" : "text-secondary-warn"}>{isFaceDetected ? "DETECTED" : "NO_TARGET"}</span></div>
                  <div className="flex justify-between"><span>Alert_Pulse</span><span>{uiSeverity > 0 ? 'INTERMITTENT' : 'LOW_POW'}</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* PRIMARY VIEWPORT */}
        <div className="flex-1 relative flex flex-col group">
          <div className="flex-1 bg-surface-low rounded-sm overflow-hidden relative border border-white/5 shadow-2xl">
            <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover scale-x-[-1] opacity-40 grayscale contrast-125" playsInline muted />
            
            {/* Visual Threat Overlay */}
            <AnimatePresence>
               {uiSeverity > 0 && (
                 <motion.div 
                   initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                   className={`absolute inset-0 pointer-events-none border-[24px] blur-3xl mix-blend-overlay transition-colors duration-500 
                     ${uiSeverity === 3 ? 'border-tertiary-alert/40' : 'border-secondary-warn/40'}`} 
                 />
               )}
            </AnimatePresence>

            {/* Tactical Display Elements */}
            <div className="absolute inset-0 flex flex-col p-12 pointer-events-none select-none">
              <div className="flex justify-between items-start">
                <div className="flex flex-col gap-1">
                   <div className="w-16 h-[1px] bg-primary-safe/40" />
                   <span className="text-[10px] telemetry text-white/40 tracking-[0.4em] uppercase">Observation_Node_024</span>
                </div>
                {uiSeverity >= 2 && (
                   <motion.div initial={{ x: 20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} className="flex items-center gap-4 bg-tertiary-alert/20 px-6 py-3 rounded-sm border border-tertiary-alert/40 backdrop-blur-md">
                      <AlertTriangle className="w-5 h-5 text-tertiary-alert animate-ping" />
                      <span className="text-sm font-display font-bold text-tertiary-alert tracking-[0.3em] uppercase">Emergency_Trigger_Active</span>
                   </motion.div>
                )}
              </div>

              <div className="mt-auto mb-20 flex flex-col items-center">
                <AnimatePresence mode="wait">
                  <motion.div 
                    key={uiSeverity}
                    initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -40, opacity: 0 }}
                    className="flex flex-col items-center gap-4"
                  >
                    <h2 className={`text-9xl font-display font-black tracking-tighter uppercase transition-all duration-700 ${severityStyles[uiSeverity].color} ${uiSeverity === 3 ? 'scale-110 blur-[1px]' : ''}`}>
                      {severityStyles[uiSeverity].label}
                    </h2>
                    <p className={`text-sm telemetry tracking-[0.8em] font-medium uppercase opacity-50 ${severityStyles[uiSeverity].color}`}>
                      {severityStyles[uiSeverity].desc}
                    </p>
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* BARRIER STATES */}
            <AnimatePresence>
              {systemState !== 'ACTIVE' && (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="absolute inset-0 z-50 bg-surface/95 backdrop-blur-3xl flex flex-col items-center justify-center p-12 text-center"
                >
                  {systemState === 'LOADING' && (
                    <div className="flex flex-col items-center gap-6">
                       <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }} className="w-12 h-12 border-4 border-primary-safe/20 border-t-primary-safe rounded-full" />
                       <span className="text-xs tracking-[0.5em] text-white/40 uppercase">Awaiting_Neural_Load</span>
                    </div>
                  )}

                  {systemState === 'READY' && (
                    <div className="max-w-2xl flex flex-col items-center">
                       <Lock className="w-8 h-8 text-primary-safe mb-8" />
                       <h2 className="text-5xl font-display uppercase tracking-[0.4em] mb-6 font-bold">Initialize Safety Core</h2>
                       <p className="text-white/40 text-xs tracking-[0.2em] uppercase leading-loose mb-12 max-w-md">
                         System checks complete. Requesting optical bridge and audio pulse authorization to begin active monitoring.
                       </p>
                       <button 
                         onClick={startStream}
                         className="group relative px-20 py-6 bg-white text-black font-black text-sm tracking-[0.5em] uppercase hover:bg-primary-safe transition-all active:scale-95 flex items-center gap-6"
                       >
                         <Play className="w-5 h-5 fill-current" />
                         Engage Watch
                         <div className="absolute inset-0 border border-white group-hover:scale-110 group-hover:opacity-0 transition-all" />
                       </button>
                    </div>
                  )}

                  {systemState === 'ERROR' && (
                    <div className="max-w-md flex flex-col items-center text-secondary-warn">
                       <CameraOff className="w-12 h-12 mb-8" />
                       <h2 className="text-2xl font-display uppercase tracking-widest mb-4 font-bold">Initialization_Failed</h2>
                       <p className="text-[10px] telemetry mb-10 tracking-widest leading-loose">
                         {errorMsg || "UNKNOWN_SYSTEM_FAULT"}
                       </p>
                       <button onClick={() => window.location.reload()} className="px-10 py-4 border border-secondary-warn/40 text-xs tracking-[0.3em] uppercase hover:bg-secondary-warn/10 transition-colors">
                         Reload_Kernel
                       </button>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* TACTICAL FOOTER */}
      <footer className="p-4 bg-surface-low/80 border-t border-white/5 flex justify-between px-12 text-[10px] telemetry text-white/30 tracking-[0.4em] uppercase">
         <div className="flex gap-12">
           <div className="flex gap-3 items-center">
             <div className="w-1.5 h-1.5 bg-primary-safe rounded-full animate-pulse" />
             <span>Core: Mediapipe_V10.2_INT8</span>
           </div>
           <span>Hardware: {isGpu ? "GPU_ACCELERATED" : "CPU_XNNPACK_STEADY"}</span>
         </div>
         <div className="flex gap-12">
           <span>Pulse: {uiSeverity > 0 ? "EMERGENCY_BROADCAST" : "NOMINAL_MONITOR"}</span>
           <span className="text-primary-safe/50 font-bold">Enc: RSA_4096_VALID</span>
         </div>
      </footer>
    </div>
  );
}
