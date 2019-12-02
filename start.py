
import socket

server_html = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_html.bind(("127.0.0.1", 8080))

server_html.listen(10)
print("http://127.0.0.1:8080/")
while True:
    conn, addr = server_html.accept()
    msg = conn.recv(1024 * 12)
    print(conn)
    # 以字节读取数据的权限去打开html_pro.html文件
    file_html = open("templates/start.html", "rb")
    # 读取文件内容
    data = file_html.read()
    # 下面这句话必须写，关于http协议的内容，以后说
    conn.sendall(bytes("HTTP/1.1 201 OK\r\n\r\n", "utf-8"))
    # 发送读取的内容
    conn.sendall(data)

    conn.close()

if __name__ == '__main__':
    main()
