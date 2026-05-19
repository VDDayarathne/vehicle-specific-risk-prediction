FROM nginx:1.27-alpine

WORKDIR /usr/share/nginx/html

# Copy the compiled/public site files into the nginx web root. The project
# places the static HTML in `frontend/public`, so copy that directory's
# contents rather than the whole frontend context to avoid serving files
# under a nested `/public` path which caused 404s for `/dashboard.html`.
COPY public/ /usr/share/nginx/html/

EXPOSE 80
