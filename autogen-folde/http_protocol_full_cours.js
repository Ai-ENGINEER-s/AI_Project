
function getDomainNameFrom(url){

const urlObject = new URL(url)
return urlObject.hostname


}

const urlOject = new URL('https://chat.openai.com/c/9730d49d-29f7-4ed0-88cf-abca321a0faf')
domaineName =getDomainNameFrom(urlOject)
console.log(`the domain name fro ${urlOject} is ${domaineName}` )