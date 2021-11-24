var nodemailer = require('nodemailer');

var transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'rohanbondre10@gmail.com',
    pass: 'Rohan@2001'
  }
});

var mailOptions = {
  from: 'rohanbondre10@gmail.com',
  to: 'rohan.bondre@walchandsangli.ac.in',
  subject: 'Sending Email using Node.js Mailer Module',
  text: 'That was easy way to send mails!'
};

transporter.sendMail(mailOptions, function(error, info){
  if (error) {
    console.log(error);
  } else {
    console.log('Email sent: ' + info.response);
  }
});
