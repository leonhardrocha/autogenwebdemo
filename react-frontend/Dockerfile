FROM node:lts-alpine
ARG DEV_MODE
ENV NODE_ENV=$DEV_MODE
WORKDIR /web

COPY . --exclude=*-lock.json ./
RUN npm set strict-ssl false
RUN npm install
# --production --silent 
# EXPOSE 3000
# EXPOSE 5173
RUN chown -R node /web
USER node
#CMD ["npm", "start"]
