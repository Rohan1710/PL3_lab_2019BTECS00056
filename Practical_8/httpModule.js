var https = require('https');

https.get('https://www.w3schools.com/nodejs/ref_https.asp',function (res) {
  res.on('data',function(data){
    console.log(data);
  })
})
