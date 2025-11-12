"use client"

import { useState, useCallback } from "react"

const API_BASE_URL = "http://localhost:5000/api"

interface ApiResponse<T = any> {
  data?: T
  error?: string
  loading: boolean
}

export function useInterviewAPI() {
  const [loading, setLoading] = useState(false)

  const apiCall = useCallback(async <T = any>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.error || "API request failed", loading: false }
      }

      return { data, loading: false }
    } catch (error) {
      return { error: `Network error: ${error}`, loading: false }
    } finally {
      setLoading(false)
    }
  }, [])

  const startInterview = useCallback(() => {
    try {
      return apiCall("/start-interview", { method: "POST" })
    } catch (error) {
      console.warn("Backend not available:", error)
      return Promise.resolve({
        data: { status: "started", message: "Interview started in demo mode" },
        loading: false,
      })
    }
  }, [apiCall])

  const executeCode = useCallback(
    (language: string, code: string) => {
      try {
        return apiCall("/execute-code", {
          method: "POST",
          body: JSON.stringify({ language, code }),
        })
      } catch (error) {
        console.warn("Backend not available:", error)
        return Promise.resolve({
          data: { output: "Code execution simulated (backend not available)" },
          loading: false,
        })
      }
    },
    [apiCall],
  )

  const generateQuestion = useCallback(
    (domain?: string, type: "coding" | "technical" = "coding") => {
      try {
        return apiCall("/generate-question", {
          method: "POST",
          body: JSON.stringify({ domain, type }),
        })
      } catch (error) {
        console.warn("Backend not available:", error)
        const sampleQuestion = `Problem: Find the two numbers in a list that add up to a target sum.

Example Input: numbers = [2, 7, 11, 15], target = 9
Example Output: [0, 1] (indices of numbers 2 and 7)

Constraints: Each input has exactly one solution, and you may not use the same element twice.`

        return Promise.resolve({
          data: { question: sampleQuestion, type: "coding" },
          loading: false,
        })
      }
    },
    [apiCall],
  )

  const evaluateSolution = useCallback(
    (code: string, question?: string) => {
      try {
        return apiCall("/evaluate-solution", {
          method: "POST",
          body: JSON.stringify({ code, question }),
        })
      } catch (error) {
        console.warn("Backend not available:", error)
        return Promise.resolve({
          data: {
            analysis: "Solution evaluation simulated (backend not available)",
            verbal_feedback: "Your solution looks good! The approach is correct and efficient.",
          },
          loading: false,
        })
      }
    },
    [apiCall],
  )

  const processSpeech = useCallback(
    (text: string) => {
      try {
        return apiCall("/process-speech", {
          method: "POST",
          body: JSON.stringify({ text }),
        })
      } catch (error) {
        console.warn("Backend not available:", error)
        return Promise.resolve({
          data: {
            response: "I understand. That's an interesting perspective. Could you elaborate more on that?",
            tone_detected: "professional",
          },
          loading: false,
        })
      }
    },
    [apiCall],
  )

  const endInterview = useCallback(() => {
    try {
      return apiCall("/end-interview", { method: "POST" })
    } catch (error) {
      console.warn("Backend not available:", error)
      return Promise.resolve({
        data: { status: "ended", message: "Interview concluded successfully" },
        loading: false,
      })
    }
  }, [apiCall])

  const getInterviewState = useCallback(() => {
    try {
      return apiCall("/get-interview-state")
    } catch (error) {
      console.warn("Backend not available:", error)
      return Promise.resolve({
        data: { active: true },
        loading: false,
      })
    }
  }, [apiCall])

  return {
    loading,
    startInterview,
    executeCode,
    generateQuestion,
    evaluateSolution,
    processSpeech,
    endInterview,
    getInterviewState,
  }
}
