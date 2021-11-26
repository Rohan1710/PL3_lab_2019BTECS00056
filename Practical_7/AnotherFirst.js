const http = require('http');
const port = 8000;

const server = http.createServer(function(req,res){
    res.write("Hello Node JS");
    res.end();
})

server.listen(port,'127.0.0.1');
