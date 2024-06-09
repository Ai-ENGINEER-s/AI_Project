// src/components/Chat.js
import React from 'react';
import { Box, Typography, TextField, IconButton, Paper } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

function Chat() {
  return (
    <Paper elevation={3} sx={{ display: 'flex', flexDirection: 'column', width: '80%', maxWidth: '600px', height: '70%', backgroundColor: '#ffffff', borderRadius: '8px' }}>
      {/* Chat Header */}
      <Box sx={{ padding: '10px 20px', borderBottom: '1px solid #e0e0e0', backgroundColor: '#f5f5f5', borderRadius: '8px 8px 0 0' }}>
        <Typography variant="h6">Chat with MySQL</Typography>
      </Box>

      {/* Chat Messages */}
      <Box sx={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
        <Typography variant="body2" color="textSecondary">Hello! I'm a SQL assistant. Ask me anything about your database.</Typography>
      </Box>

      {/* Message Input */}
      <Box sx={{ padding: '10px', borderTop: '1px solid #e0e0e0', backgroundColor: '#f5f5f5', borderRadius: '0 0 8px 8px', display: 'flex', alignItems: 'center' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type a message..."
          sx={{ marginRight: '10px', backgroundColor: '#ffffff', borderRadius: '8px' }}
        />
        <IconButton color="primary" sx={{ padding: '10px' }}>
          <SendIcon />
        </IconButton>
      </Box>
    </Paper>
  );
}

export default Chat;
