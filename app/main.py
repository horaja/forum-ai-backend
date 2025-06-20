"""
This module provides a Flask-based API for suggesting tags for forum content.

It uses a zero-shot classification model from the Hugging Face library
to classify text against a pre-defined list of computer systems concepts.
"""

import logging
from flask import Flask, request, jsonify
from transformers import pipeline
import numpy as np

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AI-Backend] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize Flask App and Classifier
app = Flask(__name__)

# TODO: For deployment, change to smaller, fine-tuned model
try:
  classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    hypothesis_template="This post is about the computer systems topic of {}."
  )
except Exception as e:
    logging.critical(f"Failed to load Hugging Face pipeline: {e}")
    # TODO: For deployment, have a fallback
    classifier = None

# TODO: For deployment, update using Admin Board
COURSE_CONCEPTS = [
  # Data Representation & Program Basics
  "Integer Representation",
  "Integer Arithmetic",
  "Floating Point Representation",
  "Floating Point Operations",
  "Bit-Level Operations",
  "Byte Ordering", # Endianness
  "String Representation",

  # Machine-Level Representation
  "Machine Code Basics", # Assembly, ISA
  "x86-64 Data Formats",
  "Registers & Operands",
  "Data Movement (mov)",
  "x86-64 Arithmetic & Logic",
  "Control Flow (Assembly)", # Jumps, Loops
  "Procedures (Assembly)", # Stack Frame, Call/Return
  "Array Allocation (Assembly)",
  "Structs & Unions (Assembly)",
  "Buffer Overflow",
  "Floating-Point Code (Assembly)",

  # Program Performance & Optimization
  "Optimizing Compilers",
  "Performance Measurement (CPE)",
  "Loop Optimization", # Code Motion, Unrolling
  "Procedure Call Optimization",
  "Memory Reference Optimization",
  "Instruction-Level Parallelism",

  # Memory Hierarchy
  "Storage Technologies", # SRAM, DRAM, Disk
  "Locality", # Temporal, Spatial
  "Cache Memories", # Organization, Mapping
  "Cache Hits & Misses",
  "Cache Write Policies",
  "Cache-Friendly Code",

  # Linking
  "Compiler Drivers",
  "Static Linking",
  "Object Files",
  "Symbol Resolution",
  "Relocation",
  "Dynamic Linking", # Shared Libraries
  "Position-Independent Code (PIC)",
  "Library Interpositioning",

  # Exceptional Control Flow (ECF)
  "Exceptions & Interrupts",
  "Exception Handling",
  "Processes", # Context, Modes
  "Context Switching",
  "System Calls",
  "Process Control (fork, execve, waitpid)",
  "Signals", # Sending, Receiving, Handling
  "Nonlocal Jumps (setjmp, longjmp)",

  # Virtual Memory (VM)
  "Physical & Virtual Addressing",
  "VM for Caching", # Page Tables, Page Faults
  "VM for Memory Management",
  "VM for Memory Protection",
  "Address Translation (MMU, TLB)",
  "Memory Mapping (mmap)",
  "Dynamic Memory Allocation", # malloc, free, Heap
  "Garbage Collection",
  "Memory Bugs", # Dangling Pointers, Leaks

  # System-Level I/O
  "Unix I/O", # Files, Descriptors
  "File Operations (Open, Close, Read, Write)",
  "Robust I/O (RIO)",
  "File Metadata",
  "File Sharing",
  "I/O Redirection",

  # Network Programming
  "Client-Server Model",
  "Networking Basics",
  "IP Addressing & DNS",
  "Sockets API", # socket, connect, bind, listen, accept
  "Host & Service Conversion",
  "Web Servers (HTTP)",

  # Concurrent Programming
  "Process-Based Concurrency",
  "Event-Based Concurrency (I/O Multiplexing)", # select, epoll
  "Thread-Based Concurrency (Pthreads)",
  "Shared Variables & Threads", # Memory Model, Mutexes
  "Semaphore Synchronization",
  "Thread Safety & Reentrancy",
  "Race Conditions",
  "Deadlocks",

  # General Programming & Debugging
  "Debugging Principles",
  "Program Profiling",
  "GDB (Debugger)",
  "Coding Style"
]

@app.route('/api/v1/suggest-tags', methods=['POST'])
def suggest_tags():
  """
  Receives text content and returns a list of suggested concept tags.

  This endpoint expects a JSON payload with a 'content' key. It uses a
  zero-shot classification model to score concepts and returns a list
  of the most relevant tags based on the Confidence Gap Cutoff Algorithm.

  Returns:
    A Flask Response object containing a JSON payload with a
    'suggested_tags' list, or an error message.
  """
  if not classifier:
    logging.error("Classifier model is not available.")
    return jsonify({"error": "Classifier service is unavailable"}), 503

  logging.info(f"Received request on {request.path} from {request.remote_addr}")

  if not request.json or 'content' not in request.json or not request.json['content']:
    logging.warning("Request received with missing or empty 'content' field.")
    return jsonify({"error": "Missing 'content' in request body"}), 400

  content = request.json['content']

  # Classification and Tag Selection
  results = classifier(content, COURSE_CONCEPTS, multi_label=True)
  '''
  Print results of ML classifier:
  logging.info(f"Raw classification results: {results}")
  '''

  # Advanced Maximum Gap Selection Algorithm
  MAX_TAGS = 3
  MIN_CONFIDENCE = 0.15

  scores = results['scores']
  labels = results['labels']

  if not scores or scores[0] < MIN_CONFIDENCE:
    suggested_tags = []
  elif len(scores) == 1:
    suggested_tags = [labels[0]]
  else:
    # Maximum Gap Selection Algorithm
    gaps = np.diff(scores) * -1
    max_gap_index = np.argmax(gaps[:MAX_TAGS])
    num_tags_to_return = max_gap_index + 1
    suggested_tags = labels[:num_tags_to_return]

  logging.info(f"Returning suggested tags: {suggested_tags}")
  return jsonify({"suggested_tags": suggested_tags})

if __name__ == '__main__':
  # TODO: In production, a WSGI server like Gunicorn should be used.
  app.run(host='0.0.0.0', port=5000, debug=True)