const userdm = require('./userDefinedModule');
const http = require('http');
const port = 3000

const server = http.createServer(function (req, res) {
  let a = 10;
  let b = 50;
  res.write("Addition of a and b: "+userdm.add(a,b));
  res.end();
})
server.listen(port,function(error){
    if(error){
        console.log("Error Occured "+ error);
    }
    else{
        console.log("Server Listening at port "+port);
    }
})
