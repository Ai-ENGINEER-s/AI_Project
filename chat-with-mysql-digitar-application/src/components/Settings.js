// src/components/Settings.js
import React from 'react';
import { TextField, Button, Typography, Box } from '@mui/material';

function Settings() {
  return (
    <Box sx={{ padding: '20px', backgroundColor: '#f0f2f5', borderRadius: '8px', height: '100%', boxShadow: 3 }}>
      <Typography variant="h6" gutterBottom>Settings</Typography>
      <Typography variant="body2" gutterBottom>This is a simple chat application using MySQL. Connect to the database and start chatting.</Typography>
      <form>
        <TextField label="Host" defaultValue="localhost" fullWidth margin="normal" variant="outlined" />
        <TextField label="Port" defaultValue="3306" fullWidth margin="normal" variant="outlined" />
        <TextField label="User" defaultValue="root" fullWidth margin="normal" variant="outlined" />
        <TextField label="Password" type="password" fullWidth margin="normal" variant="outlined" />
        <TextField label="Database" defaultValue="Chinook" fullWidth margin="normal" variant="outlined" />
        <Button variant="contained" color="primary" fullWidth sx={{ marginTop: '20px' }}>
          Connect
        </Button>
      </form>
    </Box>
  );
}

export default Settings;
