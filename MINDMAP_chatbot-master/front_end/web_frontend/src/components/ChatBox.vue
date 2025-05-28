<template>
  <div class="container-fluid vh-100 d-flex flex-column">
    <!-- Header -->
    <div class="chat-header text-white text-center py-2">
      <h2>Chatbot</h2>
    </div>

    <!-- Nội dung chat -->
    <div class="messages flex-grow-1 overflow-auto p-3" ref="messagesContainer">
      <div v-for="(msg, index) in messages" :key="index" class="message" :class="msg.sender">
        <div class="bubble" :class="{ 'full-width': msg.sender === 'bot' && msg.type === 'youtube-group' }">
          <template v-if="msg.type === 'text'">
            {{ msg.text }}
          </template>
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="input-container d-flex p-2 bg-white border-top">
      <input v-model="newMessage" @keyup.enter="sendMessage" class="form-control me-2 px-4 py-3" placeholder="Nhập tin nhắn..." />
      <button @click="sendMessage" class="btn btn-success px-4 py-3 fw-bold" :disabled="isLoading">➤</button>
    </div>
  </div>
</template>

<script>
import { ref, nextTick } from 'vue';
import { sendMessage } from '../service/chat.service.js';

export default {
  setup() {
    const messages = ref([]);
    const newMessage = ref("");
    const messagesContainer = ref(null);
    const isLoading = ref(false);

    async function sendMessageHandler() {
      if (!newMessage.value.trim() || isLoading.value) return;

      isLoading.value = true;
      messages.value.push({ text: newMessage.value, sender: "user", type: "text" });
      const userMessage = newMessage.value;
      newMessage.value = "";

      await nextTick();
      scrollToBottom();

      try {
        const response = await sendMessage(userMessage);

        // Xử lý tin nhắn phản hồi từ bot
        if (response.response) {
          messages.value.push({ text: response.response, sender: "bot", type: "text" });
        }
      } catch (error) {
        console.error("Lỗi khi gửi tin nhắn:", error);
      }

      await nextTick();
      scrollToBottom();
      isLoading.value = false;
    }

    function scrollToBottom() {
      nextTick(() => {
        if (messagesContainer.value) {
          messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
        }
      });
    }

    return { 
      messages, 
      newMessage, 
      sendMessage: sendMessageHandler, 
      messagesContainer, 
      isLoading
    };
  }
};
</script>

<style scoped>
/* Header cố định */
.chat-header {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  background: #006d08;
  z-index: 1000;
}

/* Nội dung chat */
.messages {
  margin-top: 60px;
  margin-bottom: 80px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* Định dạng tin nhắn */
.message {
  display: flex;
  align-items: center;
}

.bubble {
  padding: 10px 15px;
  border-radius: 15px;
  max-width: 70%;
  word-wrap: break-word;
}

/* Căn chỉnh tin nhắn */
.user {
  justify-content: flex-end;
  display: flex;
}

.user .bubble {
  background: #f1f1f1;
  color: black;
  border-top-right-radius: 0;
}

.bot {
  justify-content: flex-start;
  display: flex;
}

.bot .bubble {
  background: white;
  color: black;
  border-top-left-radius: 0;
}


/* Phản hồi bot toàn màn hình */
.full-width {
  width: 100%;
  max-width: none;
  display: flex;
  flex-direction: column;
}

/* Input cố định */
.input-container {
  position: fixed;
  bottom: 10px;
  left: 50%;
  width: 80%;
  height: 10%;
  transform: translateX(-50%);
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  padding: 10px;
  display: flex;
  align-items: center;
}

/* Tăng kích thước nút gửi */
.btn-lg {
  font-size: 1.5rem;
  min-width: 60px;
}
</style>
