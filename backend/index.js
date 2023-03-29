const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const msgpack = require('msgpack5')();


const app = express();
const port = 3000;

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.send({
        "message": "Hello Cyberbuddy!!"
    })
})

app.post('/api/model', async (req, res) => {
    const data = req.body.data;
    const buffer = Buffer.from(data, 'base64');
    const inputTensor = tf.tensor(msgpack.decode(buffer));

    const model = await tf.loadLayersModel('./model/solutionChallenge.h5');
    console.log('model:', model);

    try {
        const outputTensor = model.predict(inputTensor);
        const result = outputTensor.dataSync();
        console.log('result:', result);
        res.send({ output: result });
    } catch (error) {
        console.error(error);
        res.status(500).send({ error: 'Failed to run model' });
    }
});

app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});
