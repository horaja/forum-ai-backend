# Forum AI Backend Service

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

A Python microservice built with Flask and Hugging Face Transformers to provide AI-powered content analysis for a forum platform.

This service exposes a single API endpoint that accepts a block of text and returns a list of suggested topical tags, derived from a pre-defined list of concepts using a zero-shot classification model.

## Features

- **Zero-Shot Tagging:** Uses the `facebook/bart-large-mnli` model to classify content without needing a pre-trained, domain-specific dataset.
- **Custom Tag Selection Algorithm:** Implements a "Confidence Gap Cutoff" algorithm to dynamically select the most relevant tags based on the model's confidence scores.
- **Flask API:** A lightweight and simple API built with Flask, easy to run and containerize.

## API Endpoint

### `POST /api/v1/suggest-tags`

Receives text content and returns a list of suggested tags.

- **Request Body:**
  ```json
  {
    "content": "The text of the forum post, including the title and body..."
  }

- **Success Response (200 OK):**
	```json
	{
		"suggested_tags": [
    "Cache Memories",
    "Locality",
    "Cache-Friendly Code"
  	]
	}

- **Error Response (400 Bad Request):**
	```json
	{
		"error": "Missing 'content' in request body"
	}