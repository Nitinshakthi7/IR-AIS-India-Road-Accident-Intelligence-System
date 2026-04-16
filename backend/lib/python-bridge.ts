/**
 * Python Bridge - Communication layer between Next.js API routes
 * and the Python ML backend service.
 * 
 * This module provides utility functions to send requests to the Python
 * backend for model inference, training, and data processing.
 */

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || "http://localhost:8000";

interface PythonRequestOptions {
  endpoint: string;
  method?: "GET" | "POST";
  body?: Record<string, unknown>;
  timeout?: number;
}

interface PythonResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

/**
 * Send a request to the Python ML backend.
 */
export async function pythonBridge<T = unknown>(
  options: PythonRequestOptions
): Promise<PythonResponse<T>> {
  const { endpoint, method = "POST", body, timeout = 30000 } = options;

  const url = `${PYTHON_BACKEND_URL}${endpoint}`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const fetchOptions: RequestInit = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
    };

    if (body && method === "POST") {
      fetchOptions.body = JSON.stringify(body);
    }

    const response = await fetch(url, fetchOptions);
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      return {
        success: false,
        error: `Python backend error (${response.status}): ${errorText}`,
      };
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return {
        success: false,
        error: `Request to Python backend timed out after ${timeout}ms`,
      };
    }
    return {
      success: false,
      error: `Failed to connect to Python backend at ${PYTHON_BACKEND_URL}. Ensure the ML service is running.`,
    };
  }
}

/**
 * Check if the Python backend is healthy.
 */
export async function checkPythonBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return response.ok;
  } catch {
    return false;
  }
}
