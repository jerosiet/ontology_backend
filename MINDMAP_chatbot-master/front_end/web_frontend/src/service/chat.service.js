const API_URL = 'http://127.0.0.1:5000/chat';  // Đổi thành URL backend của bạn

export async function sendMessage(message) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });
        return await response.json();
    } catch (error) {
        console.error("Lỗi gửi tin nhắn:", error);
        return { response: "Lỗi kết nối với server" };
    }
}

// import axios from "axios";

// const API_URL = "http://127.0.0.1:5000/chat"; // Địa chỉ backend Flask

// export async function sendMessage(message) {
//   try {
//     const response = await axios.post(API_URL, {
//       user_id: "user123",  // Xác định user (sẽ thay thế bằng real user)
//       message: message
//     });
//     return response.data;
//   } catch (error) {
//     console.error("Lỗi khi gửi tin nhắn:", error);
//     return { response: "Có lỗi xảy ra, vui lòng thử lại sau!" };
//   }
// }