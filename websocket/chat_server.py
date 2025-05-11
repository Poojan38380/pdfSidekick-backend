import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
import json
from datetime import datetime
import uuid
from database.models import search_similar_chunks
from utils.colorLogger import print_info, print_error


class ChatManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, pdf_id: str):
        await websocket.accept()
        if pdf_id not in self.active_connections:
            self.active_connections[pdf_id] = []
        self.active_connections[pdf_id].append(websocket)
        print_info(f"New chat connection for PDF {pdf_id}")

    def disconnect(self, websocket: WebSocket, pdf_id: str):
        if pdf_id in self.active_connections:
            self.active_connections[pdf_id].remove(websocket)
            if not self.active_connections[pdf_id]:
                del self.active_connections[pdf_id]
        print_info(f"Chat connection closed for PDF {pdf_id}")

    async def broadcast_to_pdf(self, pdf_id: str, message: dict):
        if pdf_id in self.active_connections:
            for connection in self.active_connections[pdf_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print_error(
                        f"Error broadcasting message (in broadcast_to_pdf): {e}"
                    )
                    await self.disconnect(connection, pdf_id)


chat_manager = ChatManager()


async def handle_chat_message(websocket: WebSocket, pdf_id: str, message: str):
    try:
        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Send user message back to confirm receipt
        user_message = {
            "id": message_id,
            "content": message,
            "sender": "user",
            "timestamp": datetime.utcnow().isoformat(),
        }
        await websocket.send_json(user_message)

        # Search for relevant chunks
        similar_chunks = await search_similar_chunks(pdf_id, message, limit=3)

        # Generate response based on similar chunks
        if similar_chunks:
            response_content = "Based on the document, here's what I found:\n\n"
            for chunk in similar_chunks:
                response_content += f"- {chunk['content']}\n\n"
        else:
            response_content = "I couldn't find any relevant information in the document to answer your question."

        # Send assistant's response
        assistant_message = {
            "id": str(uuid.uuid4()),
            "content": response_content,
            "sender": "assistant",
            "timestamp": datetime.utcnow().isoformat(),
        }
        await websocket.send_json(assistant_message)

    except Exception as e:
        print_error(f"Error handling chat message (in handle_chat_message): {e}")
        error_message = {
            "id": str(uuid.uuid4()),
            "content": "Sorry, I encountered an error processing your message.",
            "sender": "assistant",
            "timestamp": datetime.utcnow().isoformat(),
        }
        await websocket.send_json(error_message)


async def chat_websocket_endpoint(websocket: WebSocket, pdf_id: str):
    await chat_manager.connect(websocket, pdf_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            await handle_chat_message(websocket, pdf_id, message_data["content"])
    except WebSocketDisconnect:
        print_info(f"WebSocket disconnected for PDF {pdf_id}")
        chat_manager.disconnect(websocket, pdf_id)
    except Exception as e:
        print_error(f"WebSocket error (in chat_websocket_endpoint): {e}")
        chat_manager.disconnect(websocket, pdf_id)
