from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# for deployment: change to smaller model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# for deployment: add automation to this process
# TODO: Reduce List + add to Admin board
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

@app.route('api/v1/suggest-tags', methods=['POST'])
def suggest_tags():
	"""
	Receives text content and returns a list of suggested concept tags
	"""
	if not request.json or 'content' not in request.json:
		return jsonify({"error": "Missing 'content' in request body"}), 400

	content = request.json['content']

	results = classifier(content, COURSE_CONCEPTS, multi_label=True)

	confidence_threshold = 0.80
	suggested_tags = [
		results['labels'][i] for i, score in enumerate(results['scores']) if score > confidence_threshold
	]

	return jsonify({"suggested_tags": suggested_tags})

if __name__ == '__main__':
	# for local development only:
	app.run(host='0.0.0.0', port=5000, debug=True)