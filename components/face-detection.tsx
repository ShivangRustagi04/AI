"use client"
import { useEffect, useRef } from "react"

declare global {
  interface Window {
    FaceMesh?: any
    Camera?: any
  }
}

export default function FaceDetection() {
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const loadScripts = async () => {
        try {
          // Dynamically load required scripts
          await Promise.all([
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js'),
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js')
          ])

          if (videoRef.current && window.FaceMesh && window.Camera) {
            const faceMesh = new window.FaceMesh({
              locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
            })

            faceMesh.setOptions({
              maxNumFaces: 1,
              refineLandmarks: true,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5
            })

            faceMesh.onResults(async (results: any) => {
              const facePresent = results.multiFaceLandmarks?.length > 0
              let gazeAway = false

              if (facePresent) {
                const landmarks = results.multiFaceLandmarks[0]
                const leftEye = landmarks[33]
                const rightEye = landmarks[263]

                const dx = rightEye.x - leftEye.x
                const dy = rightEye.y - leftEye.y
                const angle = Math.atan2(dy, dx) * 180 / Math.PI

                gazeAway = Math.abs(angle) > 20
              }

              try {
                await fetch("http://localhost:5000/api/face-status", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    face_present: facePresent,
                    gaze_away: gazeAway
                  })
                })
              } catch (error) {
                console.error("Error sending face status:", error)
              }
            })

            const camera = new window.Camera(videoRef.current, {
              onFrame: async () => {
                await faceMesh.send({ image: videoRef.current! })
              },
              width: 640,
              height: 480
            })
            camera.start()
          }
        } catch (error) {
          console.error("Error loading face detection:", error)
        }
      }

      loadScripts()
    }

    return () => {
      // Cleanup if needed
    }
  }, [])

  const loadScript = (src: string) => {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve(true)
        return
      }

      const script = document.createElement('script')
      script.src = src
      script.onload = () => resolve(true)
      script.onerror = () => reject(new Error(`Script load error for ${src}`))
      document.body.appendChild(script)
    })
  }

  return (
    <video 
      ref={videoRef} 
      autoPlay 
      muted 
      playsInline 
      className="hidden" // Hidden since we're using it for detection only
    />
  )
}