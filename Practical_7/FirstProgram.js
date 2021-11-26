const http = require('http');
const port = 3000;

const server = http.createServer(function(req,res){
    res.write("Hello Node JS");
    res.end();
})

server.listen(port,function(error){
    if(error){
        console.log('Something went wrong',error);
    }
    else{
        console.log('Server is listening',port);
    }
})
