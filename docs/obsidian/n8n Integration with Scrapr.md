# n8n Integration with Scrapr

## Overview
Scrapr can be integrated with n8n to create powerful document processing workflows. This guide shows how to set up and use Scrapr with n8n.

## Prerequisites
- Scrapr API running (Docker or local)
- n8n instance accessible
- Network connectivity between services

## Basic Integration

### 1. Single Document Processing
```json
{
  "name": "Document Extraction Workflow",
  "nodes": [
    {
      "parameters": {
        "path": "/watch/documents",
        "event": "add"
      },
      "name": "Watch Folder",
      "type": "n8n-nodes-base.localFileTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://scrapr:8000/extract",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "file",
              "parameterType": "formBinaryData",
              "inputDataFieldName": "data"
            }
          ]
        },
        "queryParameters": {
          "parameters": [
            {
              "name": "enable_ocr",
              "value": "true"
            },
            {
              "name": "output_format",
              "value": "markdown"
            }
          ]
        }
      },
      "name": "Extract with Scrapr",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    }
  ]
}
```